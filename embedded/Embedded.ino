 #include <ESP8266WiFi.h>
#include <TFT_eSPI.h>
#include <SPI.h>
#include <time.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include <math.h>
#include <DHT.h>

// ---------- WIFI (EDIT THESE!) ----------
const char* WIFI_SSID     = "iphone(2)";
const char* WIFI_PASSWORD = "Espoir%55";

// ---------- Backend ----------
// These endpoints are intentionally public (no auth) so the firmware can
// send + read data without a token.
const char* BACKEND_URL       = "https://microweather-forecast-ml-backend.onrender.com/predict";
const char* MEASUREMENTS_URL  = "https://microweather-forecast-ml-backend.onrender.com/measurements/latest";
const char* HEALTH_URL        = "https://microweather-forecast-ml-backend.onrender.com/health";

// ---------- NTP / Time ----------
const char* NTP_SERVER = "pool.ntp.org";
// Rwanda = UTC+2
const long GMT_OFFSET_SEC      = 2 * 3600;
const int  DAYLIGHT_OFFSET_SEC = 0;

// ---------- Objects ----------
TFT_eSPI tft = TFT_eSPI();    // Uses your User_Setup.h (ST7789 240x280)

// ---------- DHT11 (Temp & Humidity) ----------
#define DHTPIN  4       // D2 — free now that BME280 is gone
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// For periodic updates
unsigned long lastSensorUpdate = 0;
unsigned long lastClockUpdate  = 0;
unsigned long lastWifiCheck    = 0;
float currentTemp  = 0;
float currentHum   = 0;
float currentPress = 0;

// Forecast state
float predictedRain  = 0;
float predictedTemp  = 0;
float predictedHum   = 0;
float predictedPress = 0;
String predictedTime = "";
bool haveForecast    = false;

// Use simulated rainfall values while no physical sensor is available
float lastRainfall = 0.0f;

// Latest readings synced from the backend (used as fallback when the
// DHT11 fails a read)
float backendTemp  = 0;
float backendHum   = 0;
float backendPress = 0;
bool  haveBackendRead = false;
unsigned long lastMeasurementFetch = 0;

// ----------------------------------------------------------
// Layout constants tuned for a 240 x 280 portrait display
// (the original layout assumed a ~320px tall panel; every
// y-coordinate below has been rescaled so nothing is clipped)
// ----------------------------------------------------------
#define TOP_BAR_H      32   // date bar: 0..32
#define SEP2_Y         204  // separator between clock area and tiles

#define CLOCK_Y        42
#define INDOOR_Y       88
#define MAIN_TEMP_Y    114
#define COND_Y         150

#define RAIN_AREA_Y    172
#define RAIN_AREA_H    32
#define RAIN_BADGE_H   26
#define RAIN_BADGE_W   180

#define TITLE_Y        207  // "Temp/Humid/Press" column headers (static)
#define VALUE_Y        218  // live numeric readout row
#define FORECAST_LABEL_Y 238
#define FORECAST_CARD_Y  248
#define FORECAST_CARD_H  28

// ----------------------------------------------------------
// Visual helpers
// ----------------------------------------------------------
void drawRainIcon(int cx, int cy, uint16_t color) {
  tft.fillCircle(cx, cy - 4, 6, color);
  tft.fillTriangle(cx - 6, cy - 2, cx + 6, cy - 2, cx, cy + 8, color);
}

void drawThermometerIcon(int x, int y, uint16_t color) {
  tft.drawRoundRect(x, y - 10, 8, 20, 3, color);
  tft.fillCircle(x + 4, y + 6, 6, color);
}

void drawHumidityIcon(int cx, int cy, uint16_t color) {
  tft.fillCircle(cx, cy - 2, 6, color);
  tft.fillTriangle(cx - 6, cy + 2, cx + 6, cy + 2, cx, cy + 10, color);
}

void drawPressureIcon(int cx, int cy, uint16_t color) {
  tft.drawCircle(cx, cy, 8, color);
  tft.drawLine(cx, cy, cx, cy - 5, color);
  tft.drawLine(cx, cy, cx + 4, cy + 3, color);
}

// ----------------------------------------------------------
// Forecast card rendering
// ----------------------------------------------------------
void drawForecastCard(int columnIndex, const char* title, float value, const char* unit, void (*iconFn)(int, int, uint16_t)) {
  int colWidth = tft.width() / 3;
  int cardWidth = colWidth - 12;
  int cardHeight = FORECAST_CARD_H;
  int x = columnIndex * colWidth + 6;
  int y = FORECAST_CARD_Y;

  tft.fillRoundRect(x, y, cardWidth, cardHeight, 6, TFT_NAVY);
  tft.drawRoundRect(x, y, cardWidth, cardHeight, 6, TFT_CYAN);

  if (iconFn) {
    iconFn(x + 11, y + cardHeight / 2, TFT_CYAN);
  }

  tft.setTextSize(1);
  tft.setTextColor(TFT_WHITE, TFT_NAVY);
  tft.setCursor(x + 22, y + 3);
  tft.print(title);
  tft.setCursor(x + 22, y + 16);
  tft.printf("%.2f%s", value, unit);
}

void drawRainBadge() {
  int badgeWidth = RAIN_BADGE_W;
  int badgeHeight = RAIN_BADGE_H;
  int x = (tft.width() - badgeWidth) / 2;
  int y = RAIN_AREA_Y;

  tft.fillRoundRect(x, y, badgeWidth, badgeHeight, 8, TFT_NAVY);
  tft.drawRoundRect(x, y, badgeWidth, badgeHeight, 8, TFT_CYAN);
  drawRainIcon(x + 18, y + badgeHeight / 2 + 4, TFT_CYAN);

  tft.setTextSize(1);
  tft.setTextColor(TFT_WHITE, TFT_NAVY);
  tft.setCursor(x + 36, y + 5);
  tft.print("Next Rain");
  tft.setCursor(x + 36, y + 15);
  tft.printf("%.3f mm", predictedRain);

  if (predictedTime.length() >= 16) {
    String hhmm = predictedTime.substring(11, 16);
    tft.setCursor(x + badgeWidth - 50, y + 5);
    tft.print(hhmm);
  }
}

// ----------------------------------------------------------
// Random rainfall simulation
// ----------------------------------------------------------
float generatePseudoRainfall() {
  float base = random(0, 2000) / 2000.0f;  // 0.000 - 0.999 steps
  float rainfall = base * base * 2.0f;     // bias towards small values, max ~2.0
  rainfall = constrain(rainfall, 0.0f, 2.0f);
  rainfall = roundf(rainfall * 1000.0f) / 1000.0f;
  return rainfall;
}

// ----------------------------------------------------------
// Helper to build ISO timestamp with configured timezone
// ----------------------------------------------------------
String buildIsoTimestamp(const struct tm& timeinfo) {
  char tsBuf[32];
  strftime(tsBuf, sizeof(tsBuf), "%Y-%m-%dT%H:%M:%S", &timeinfo);

  long offsetSeconds = GMT_OFFSET_SEC + DAYLIGHT_OFFSET_SEC;
  int offsetHours = offsetSeconds / 3600;
  int offsetMinutes = abs((offsetSeconds % 3600) / 60);

  char offsetBuf[8];
  snprintf(offsetBuf, sizeof(offsetBuf), "%c%02d:%02d",
           (offsetSeconds >= 0) ? '+' : '-',
           abs(offsetHours), offsetMinutes);

  String iso = String(tsBuf);
  iso += offsetBuf;
  return iso;
}

// ----------------------------------------------------------
// Call backend for forecast
// ----------------------------------------------------------
bool fetchForecastFromBackend() {
  if (WiFi.status() != WL_CONNECTED) {
    ensureWiFiReconnect();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi not connected. Skipping forecast fetch.");
      return false;
    }
  }

  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("Failed to obtain time for forecast payload.");
    return false;
  }

  String isoTimestamp = buildIsoTimestamp(timeinfo);

  float simulatedRain = generatePseudoRainfall();
  lastRainfall = simulatedRain;

  DynamicJsonDocument payload(512);
  JsonArray samples = payload.createNestedArray("samples");
  JsonObject sample = samples.createNestedObject();
  sample["Timestamp"] = isoTimestamp;
  sample["previous_rainfall"] = simulatedRain;
  sample["previous_pressure"] = currentPress;
  sample["previous_temperature"] = currentTemp;
  sample["previous_humidity"] = currentHum;

  String jsonBody;
  serializeJson(payload, jsonBody);

  // Render free instances sleep and cold-start slowly, so the very first
  // request can hit a dead TLS connection (HTTP -5 = connection lost).
  // Wake the backend up with a cheap /health GET before posting.
  if (!wakeBackend()) {
    Serial.println("Backend unreachable (GET /health failed). Skipping forecast fetch.");
    return false;
  }

  // Retry the POST a few times so a slow cold start doesn't count as an
  // error. Transport errors (negative codes) are ESP8266/TLS issues;
  // positive codes mean the server answered (check Render logs then).
  for (int attempt = 1; attempt <= 3; attempt++) {
    WiFiClientSecure client;
    client.setInsecure();
    // Generous BearSSL buffers so the full forecast response can be read
    // without the TLS connection being dropped mid-transfer (HTTP -5).
    client.setBufferSizes(2048, 2048);

    HTTPClient http;
    http.setTimeout(12000);
    http.begin(client, BACKEND_URL);
    http.addHeader("Content-Type", "application/json");

    int statusCode = http.POST(jsonBody);

    if (statusCode >= 200 && statusCode < 300) {
      String response = http.getString();
      http.end();
      return parseForecastResponse(response);
    }

    String errDetails = (statusCode < 0) ? http.errorToString(statusCode) : "";
    http.end();

    if (statusCode < 0) {
      Serial.printf("Forecast fetch attempt %d/3 failed (HTTP %d: %s)\n",
                    attempt, statusCode, errDetails.c_str());
    } else {
      // Reached the server but it answered with an error -> backend-side
      // problem (check Render logs / MongoDB config), not the board.
      Serial.printf("Forecast request failed: HTTP %d\n", statusCode);
    }

    delay(1500);
  }

  return false;
}

// ----------------------------------------------------------
// Wake a sleeping backend instance (Render free tier) and verify
// it is actually responding before we spend time posting forecasts.
// ----------------------------------------------------------
bool wakeBackend() {
  if (WiFi.status() != WL_CONNECTED) return false;

  WiFiClientSecure client;
  client.setInsecure();
  client.setBufferSizes(1024, 512);

  HTTPClient http;
  http.setTimeout(10000);
  http.begin(client, HEALTH_URL);

  int statusCode = http.GET();
  http.end();
  return (statusCode >= 200 && statusCode < 300);
}

// ----------------------------------------------------------
// Parse the /predict response body into the forecast state.
// ----------------------------------------------------------
bool parseForecastResponse(const String& response) {
  DynamicJsonDocument respDoc(4096);
  DeserializationError err = deserializeJson(respDoc, response);
  if (err) {
    Serial.print("Failed to parse forecast response: ");
    Serial.println(err.c_str());
    return false;
  }

  JsonArray items = respDoc["items"].as<JsonArray>();
  if (items.isNull() || items.size() == 0) {
    Serial.println("Forecast response missing items array.");
    return false;
  }

  JsonObject first = items[0];
  JsonObject predicted = first["predicted"];
  if (predicted.isNull()) {
    Serial.println("Forecast response missing predicted block.");
    return false;
  }

  float parsedRain = predicted["rainfall"] | predictedRain;
  predictedRain  = fabsf(parsedRain);
  predictedPress = predicted["pressure"]    | currentPress;
  predictedTemp  = predicted["temperature"] | currentTemp;
  predictedHum   = predicted["humidity"]    | currentHum;
  predictedTime  = first["Timestamp"].as<const char*>();

  lastRainfall = predictedRain; // feed next request
  haveForecast = true;

  Serial.println("Forecast updated successfully.");
  return true;
}

// ----------------------------------------------------------
// Keep a decent WiFi link. The board auto-reconnects instead of
// silently skipping requests when the router drops the link.
// ----------------------------------------------------------
void ensureWiFiReconnect() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.println("WiFi disconnected - reconnecting...");
  WiFi.disconnect();
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  for (int attempt = 0; attempt < 20 && WiFi.status() != WL_CONNECTED; attempt++) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi reconnected");
    // Clock may have drifted while offline; re-sync with NTP
    configTime(GMT_OFFSET_SEC, DAYLIGHT_OFFSET_SEC, NTP_SERVER);
    updateDate();
  } else {
    Serial.println("\nWiFi reconnect failed - will retry");
  }
}

// ----------------------------------------------------------
// Read the latest measurement stored on the backend (GET).
// The value is used as a live fallback whenever the local DHT11
// returns NaN, so the display never shows stale/broken numbers.
// ----------------------------------------------------------
bool fetchLatestMeasurements() {
  if (WiFi.status() != WL_CONNECTED) {
    ensureWiFiReconnect();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi not connected. Skipping measurements fetch.");
      return false;
    }
  }

  WiFiClientSecure client;
  client.setInsecure();
  client.setBufferSizes(1024, 512);

  HTTPClient http;
  http.setTimeout(8000);
  http.begin(client, MEASUREMENTS_URL);

  int statusCode = http.GET();
  if (statusCode < 200 || statusCode >= 300) {
    Serial.printf("Measurements request failed: HTTP %d\n", statusCode);
    http.end();
    return false;
  }

  String response = http.getString();
  http.end();

  DynamicJsonDocument doc(512);
  DeserializationError err = deserializeJson(doc, response);
  if (err) {
    Serial.print("Failed to parse measurements response: ");
    Serial.println(err.c_str());
    return false;
  }

  JsonObject values = doc["values"];
  if (values.isNull()) {
    Serial.println("Measurements response missing values block.");
    return false;
  }

  backendTemp  = values["temperature"] | currentTemp;
  backendHum   = values["humidity"]    | currentHum;
  backendPress = values["pressure"]    | currentPress;
  haveBackendRead = true;

  Serial.printf("Backend sync OK (temp %.1f, hum %.1f, press %.0f)\n",
                backendTemp, backendHum, backendPress);
  return true;
}

// ==========================================================
// Helper: draw text centered horizontally (compatible version)
// ==========================================================
void drawCenteredText(const char* text, int y, int fontSize, uint16_t color) {
  tft.setTextSize(fontSize);
  tft.setTextColor(color, TFT_BLACK);

  int len = strlen(text);
  // default 5x7 font uses ~6 pixels per character (5 glyph + 1 space)
  int textWidth = len * 6 * fontSize;
  int x = (tft.width() - textWidth) / 2;

  tft.setCursor(x, y);
  tft.print(text);
}

// ==========================================================
// Draw static layout (boxes, separators, labels)
// ==========================================================
void drawStaticLayout() {
  tft.fillScreen(TFT_BLACK);

  // Top bar (date)
  tft.fillRect(0, 0, tft.width(), TOP_BAR_H, TFT_BLACK);
  tft.drawFastHLine(0, TOP_BAR_H, tft.width(), TFT_DARKGREY);

  // Separator under main clock area
  tft.drawFastHLine(0, SEP2_Y, tft.width(), TFT_DARKGREY);

  // Bottom background
  tft.fillRect(0, SEP2_Y + 1, tft.width(), tft.height() - (SEP2_Y + 1), TFT_BLACK);

  // Vertical lines for 3 columns
  int colWidth = tft.width() / 3;
  tft.drawFastVLine(colWidth,       SEP2_Y + 1, tft.height() - (SEP2_Y + 1), TFT_DARKGREY);
  tft.drawFastVLine(colWidth * 2,   SEP2_Y + 1, tft.height() - (SEP2_Y + 1), TFT_DARKGREY);

  // Titles in each bottom tile (small font so it fits the shorter panel)
  tft.setTextSize(1);
  tft.setTextColor(TFT_YELLOW, TFT_BLACK);

  const char* titles[3] = {"Temp", "Humid", "Press"};
  for (int i = 0; i < 3; i++) {
    int textWidth = strlen(titles[i]) * 6;
    int x = i * colWidth + (colWidth - textWidth) / 2;
    tft.setCursor(x, TITLE_Y);
    tft.print(titles[i]);
  }
}

// ==========================================================
// Update date (top bar)
// ==========================================================
void updateDate() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return;

  char buf[40];
  // Example: MON AUG 09 2021
  strftime(buf, sizeof(buf), "%a %b %d %Y", &timeinfo);

  tft.fillRect(0, 0, tft.width(), TOP_BAR_H, TFT_BLACK);
  drawCenteredText(buf, 8, 2, TFT_WHITE);
}

// ==========================================================
// Update main clock + big temperature
// ==========================================================
void updateClock() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return;

  char timeBuf[16];
  strftime(timeBuf, sizeof(timeBuf), "%H:%M:%S", &timeinfo);

  // Clear middle area (clock & big temp)
  tft.fillRect(0, TOP_BAR_H + 1, tft.width(), SEP2_Y - (TOP_BAR_H + 1), TFT_BLACK);

  // Big clock
  drawCenteredText(timeBuf, CLOCK_Y, 3, TFT_WHITE);

  // Location / label
  drawCenteredText("Indoor", INDOOR_Y, 2, TFT_CYAN);

  // Main temperature
  char tempBuf[16];
  snprintf(tempBuf, sizeof(tempBuf), "%.1fC", currentTemp);
  drawCenteredText(tempBuf, MAIN_TEMP_Y, 3, TFT_WHITE);

  // Simple condition based on humidity
  const char* cond = "Comfort";
  if (currentHum > 80) cond = "Humid";
  else if (currentHum < 30) cond = "Dry";

  drawCenteredText(cond, COND_Y, 2, TFT_LIGHTGREY);

  tft.fillRect(0, RAIN_AREA_Y, tft.width(), RAIN_AREA_H, TFT_BLACK);
  if (haveForecast) {
    drawRainBadge();
  } else {
    drawCenteredText("Forecast pending...", RAIN_AREA_Y + 6, 1, TFT_DARKGREY);
  }
}

// ==========================================================
// Update bottom tiles (Temp / Humid / Press)
// ==========================================================
void updateSensorTiles() {
  int colWidth = tft.width() / 3;

  // Clear numeric + forecast area (titles above VALUE_Y stay static)
  tft.fillRect(0, VALUE_Y - 2, tft.width(), tft.height() - (VALUE_Y - 2), TFT_BLACK);

  tft.setTextSize(2);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);

  char buf[16];

  // Temperature
  snprintf(buf, sizeof(buf), "%.1fC", currentTemp);
  tft.setCursor(colWidth / 2 - (int)(strlen(buf) * 6), VALUE_Y);
  tft.print(buf);

  // Humidity
  snprintf(buf, sizeof(buf), "%.1f%%", currentHum);
  tft.setCursor(colWidth + colWidth / 2 - (int)(strlen(buf) * 6), VALUE_Y);
  tft.print(buf);

  // Pressure (unit omitted here to fit the narrow column; "Press" title covers it)
  snprintf(buf, sizeof(buf), "%.0f", currentPress);
  tft.setCursor(colWidth * 2 + colWidth / 2 - (int)(strlen(buf) * 6), VALUE_Y);
  tft.print(buf);

  if (haveForecast) {
    String label = "Predicted next interval";
    if (predictedTime.length() >= 16) {
      label += " @ ";
      label += predictedTime.substring(11, 16);
    }
    drawCenteredText(label.c_str(), FORECAST_LABEL_Y, 1, TFT_CYAN);

    drawForecastCard(0, "Temp", predictedTemp, "C", drawThermometerIcon);
    drawForecastCard(1, "Humid", predictedHum, "%", drawHumidityIcon);
    drawForecastCard(2, "Press", predictedPress, "", drawPressureIcon);
  } else {
    drawCenteredText("Predicted data unavailable", FORECAST_LABEL_Y, 1, TFT_DARKGREY);
  }
}

// ==========================================================
// Read sensors: real DHT11 for Temp & Humidity, simulated
// random-walk value for Pressure (no barometer attached).
// ==========================================================
void readSensors() {
  // ---- DHT11: Temp & Humidity ----
  float t = dht.readTemperature();
  float h = dht.readHumidity();

  // DHT11 occasionally returns NaN on a failed read (it's a slow,
  // finicky sensor — 1 reading/sec max). On failure, fall back to the
  // last measurement synced from the backend, else keep the last value.
  if (!isnan(t)) {
    currentTemp = t;
  } else if (haveBackendRead) {
    Serial.println("DHT11 read failed (temperature) - using backend reading");
    currentTemp = backendTemp;
  } else {
    Serial.println("DHT11 read failed (temperature) - keeping last value");
  }

  if (!isnan(h)) {
    currentHum = h;
  } else if (haveBackendRead) {
    Serial.println("DHT11 read failed (humidity) - using backend reading");
    currentHum = backendHum;
  } else {
    Serial.println("DHT11 read failed (humidity) - keeping last value");
  }

  // ---- Simulated Pressure (no physical barometer) ----
  static bool pressureSeeded = false;
  if (!pressureSeeded) {
    currentPress = 1013.0f;
    pressureSeeded = true;
  }
  // random(-10,11)/10.0 -> a step of -1.0 .. +1.0
  currentPress += random(-10, 11) / 10.0f;
  currentPress = constrain(currentPress, 990.0f, 1030.0f);
}

// ==========================================================
// SETUP
// ==========================================================
void setup() {
  Serial.begin(115200);

  // ESP8266 has no esp_random(); seed from the boot-time cycle counter
  // and a floating analog pin instead.
  randomSeed(ESP.getCycleCount() ^ analogRead(A0));

  // DHT11
  dht.begin();

  // TFT
  tft.init();
  tft.setRotation(0);                 // Portrait: 240 wide x 280 tall
  drawStaticLayout();

  // WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  // Time via NTP
  configTime(GMT_OFFSET_SEC, DAYLIGHT_OFFSET_SEC, NTP_SERVER);
  delay(1000);
  updateDate();

  // First sensor read & initial display
  // Give the DHT11 a moment after power-up before its first read;
  // it's unreliable if read too soon after begin().
  delay(1500);
  readSensors();
  updateClock();
  updateSensorTiles();
}

// ==========================================================
// LOOP
// ==========================================================
void loop() {
  unsigned long now = millis();

  // Keep the WiFi link alive (cheap check each loop)
  if (WiFi.status() != WL_CONNECTED && (now - lastWifiCheck >= 10000)) {
    lastWifiCheck = now;
    ensureWiFiReconnect();
  }

  // Clock every 1 s
  if (now - lastClockUpdate >= 1000) {
    lastClockUpdate = now;
    updateClock();
  }

  // Sensor every 5 s (well above DHT11's 1 Hz limit)
  if (now - lastSensorUpdate >= 5000) {
    lastSensorUpdate = now;
    readSensors();
    updateSensorTiles();

    if (fetchForecastFromBackend()) {
      updateSensorTiles();
    }

    // Refresh the stored backend measurement every ~30 s so the
    // device always has a recent server-side fallback reading.
    if (now - lastMeasurementFetch >= 30000) {
      lastMeasurementFetch = now;
      fetchLatestMeasurements();
    }
  }

  // Date roughly every minute
  static int lastMinute = -1;
  struct tm timeinfo;
  if (getLocalTime(&timeinfo)) {
    if (timeinfo.tm_min != lastMinute) {
      lastMinute = timeinfo.tm_min;
      updateDate();
    }
  }
}