#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_BME280.h>
#include <TFT_eSPI.h>
#include <SPI.h>
#include <time.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ---------- WIFI (EDIT THESE!) ----------
const char* WIFI_SSID     = "UPI";
const char* WIFI_PASSWORD = "";

// ---------- Backend ----------
const char* BACKEND_URL = "https://microweather-forecast-ml-backend.onrender.com/predict";

// ---------- NTP / Time ----------
const char* NTP_SERVER = "pool.ntp.org";
// Rwanda = UTC+2
const long GMT_OFFSET_SEC      = 2 * 3600;
const int  DAYLIGHT_OFFSET_SEC = 0;

// ---------- Objects ----------
TFT_eSPI tft = TFT_eSPI();    // Uses your User_Setup.h
Adafruit_BME280 bme;

#define SDA_PIN 21
#define SCL_PIN 22

// For periodic updates
unsigned long lastSensorUpdate = 0;
unsigned long lastClockUpdate  = 0;
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

// Use last rainfall value as lag for next request (ESP lacks rainfall sensor)
float lastRainfall = 0.0f;

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
    Serial.println("WiFi not connected. Skipping forecast fetch.");
    return false;
  }

  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("Failed to obtain time for forecast payload.");
    return false;
  }

  String isoTimestamp = buildIsoTimestamp(timeinfo);

  DynamicJsonDocument payload(512);
  JsonArray samples = payload.createNestedArray("samples");
  JsonObject sample = samples.createNestedObject();
  sample["Timestamp"] = isoTimestamp;
  sample["previous_rainfall"] = lastRainfall;
  sample["previous_pressure"] = currentPress;
  sample["previous_temperature"] = currentTemp;
  sample["previous_humidity"] = currentHum;

  String jsonBody;
  serializeJson(payload, jsonBody);

  HTTPClient http;
  http.setTimeout(8000);
  http.begin(BACKEND_URL);
  http.addHeader("Content-Type", "application/json");

  int statusCode = http.POST(jsonBody);
  if (statusCode < 200 || statusCode >= 300) {
    Serial.printf("Forecast request failed: HTTP %d\n", statusCode);
    http.end();
    return false;
  }

  String response = http.getString();
  http.end();

  DynamicJsonDocument respDoc(2048);
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

  predictedRain  = predicted["rainfall"]    | predictedRain;
  predictedPress = predicted["pressure"]    | currentPress;
  predictedTemp  = predicted["temperature"] | currentTemp;
  predictedHum   = predicted["humidity"]    | currentHum;
  predictedTime  = first["Timestamp"].as<const char*>();

  lastRainfall = predictedRain; // feed next request
  haveForecast = true;

  Serial.println("Forecast updated successfully.");
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
  tft.fillRect(0, 0, tft.width(), 40, TFT_BLACK);
  tft.drawFastHLine(0, 40, tft.width(), TFT_DARKGREY);

  // Separator under main area
  tft.drawFastHLine(0, 220, tft.width(), TFT_DARKGREY);

  // Bottom background
  tft.fillRect(0, 221, tft.width(), tft.height() - 221, TFT_BLACK);

  // Vertical lines for 3 columns
  int colWidth = tft.width() / 3;
  tft.drawFastVLine(colWidth,       221, tft.height() - 221, TFT_DARKGREY);
  tft.drawFastVLine(colWidth * 2,   221, tft.height() - 221, TFT_DARKGREY);

  // Titles in each bottom tile
  tft.setTextSize(2);
  tft.setTextColor(TFT_YELLOW, TFT_BLACK);

  tft.setCursor(colWidth/2 - 30, 230);
  tft.print("Temp");

  tft.setCursor(colWidth + colWidth/2 - 40, 230);
  tft.print("Humid");

  tft.setCursor(colWidth*2 + colWidth/2 - 40, 230);
  tft.print("Press");
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

  tft.fillRect(0, 0, tft.width(), 40, TFT_BLACK);
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
  tft.fillRect(0, 40, tft.width(), 180, TFT_BLACK);

  // Big clock
  drawCenteredText(timeBuf, 60, 4, TFT_WHITE);

  // Location / label
  drawCenteredText("Indoor", 110, 2, TFT_CYAN);

  // Main temperature
  char tempBuf[16];
  snprintf(tempBuf, sizeof(tempBuf), "%.1fC", currentTemp);
  drawCenteredText(tempBuf, 140, 4, TFT_WHITE);

  // Simple condition based on humidity
  const char* cond = "Comfort";
  if (currentHum > 80) cond = "Humid";
  else if (currentHum < 30) cond = "Dry";

  drawCenteredText(cond, 180, 2, TFT_LIGHTGREY);

  tft.fillRect(0, 200, tft.width(), 20, TFT_BLACK);
  if (haveForecast) {
    char forecastBuf[40];
    snprintf(forecastBuf, sizeof(forecastBuf), "Next Rain: %.2f mm", predictedRain);
    drawCenteredText(forecastBuf, 200, 1, TFT_CYAN);
  } else {
    drawCenteredText("Forecast pending...", 200, 1, TFT_DARKGREY);
  }
}

// ==========================================================
// Update bottom tiles (Temp / Humid / Press)
// ==========================================================
void updateSensorTiles() {
  int colWidth = tft.width() / 3;

  // Clear numeric area
  tft.fillRect(0, 260, tft.width(), tft.height() - 260, TFT_BLACK);

  tft.setTextSize(2);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);

  // Temperature
  tft.setCursor(colWidth/2 - 40, 270);
  tft.printf("%.1fC", currentTemp);

  // Humidity
  tft.setCursor(colWidth + colWidth/2 - 40, 270);
  tft.printf("%.1f%%", currentHum);

  // Pressure
  tft.setCursor(colWidth*2 + colWidth/2 - 52, 270);
  tft.printf("%.1f", currentPress);
  tft.setTextSize(1);
  tft.print(" hPa");

  if (haveForecast) {
    tft.setTextSize(1);
    tft.setTextColor(TFT_CYAN, TFT_BLACK);

    tft.setCursor(colWidth/2 - 40, 295);
    tft.printf("-> %.1fC", predictedTemp);

    tft.setCursor(colWidth + colWidth/2 - 40, 295);
    tft.printf("-> %.1f%%", predictedHum);

    tft.setCursor(colWidth*2 + colWidth/2 - 52, 295);
    tft.printf("-> %.1f hPa", predictedPress);
  }
}

// ==========================================================
// Read BME280 sensor
// ==========================================================
void readBME280() {
  currentTemp  = bme.readTemperature();
  currentHum   = bme.readHumidity();
  currentPress = bme.readPressure() / 100.0F; // Pa -> hPa
}

// ==========================================================
// SETUP
// ==========================================================
void setup() {
  Serial.begin(115200);

  // I2C + BME280
  Wire.begin(SDA_PIN, SCL_PIN);
  bool ok = bme.begin(0x76);          // most boards use 0x76
  if (!ok) ok = bme.begin(0x77);      // some use 0x77

  if (!ok) {
    Serial.println("BME280 not found!");
  } else {
    Serial.println("BME280 OK");
  }

  // TFT
  tft.init();
  tft.setRotation(0);                 // Portrait like your example
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
  readBME280();
  updateClock();
  updateSensorTiles();
}

// ==========================================================
// LOOP
// ==========================================================
void loop() {
  unsigned long now = millis();

  // Clock every 1 s
  if (now - lastClockUpdate >= 1000) {
    lastClockUpdate = now;
    updateClock();
  }

  // Sensor every 5 s
  if (now - lastSensorUpdate >= 5000) {
    lastSensorUpdate = now;
    readBME280();
    updateSensorTiles();

    if (fetchForecastFromBackend()) {
      updateSensorTiles();
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
