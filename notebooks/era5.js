
// ERA5 MONTHLY PRECIPITATION & TEMPERATURE DOWNLOAD (1995-2025)
// Predictor inputs for RBFN gap-filling — available for ALL months

// Requires: aoi already imported as an asset in this script.

var startYear = 1995;
var endYear = 2025;

// ERA5 Monthly Aggregates (0.25°) - has precipitation and 2m temperature
var era5Monthly = ee.ImageCollection('ECMWF/ERA5/MONTHLY');

// ----- EXPORT LOOP -----
for (var year = startYear; year <= endYear; year++) {
  for (var month = 1; month <= 12; month++) {
    (function(y, m) {
      var startDate = ee.Date.fromYMD(y, m, 1);
      var endDate = startDate.advance(1, 'month');

      var monthImg = era5Monthly
        .filterDate(startDate, endDate)
        .first(); // one image per month already in this collection

      var mm = m < 10 ? '0' + m : '' + m;

      // ----- PRECIPITATION -----
      // band: total_precipitation (m/day, accumulated monthly mean) - convert to mm
      var precip = ee.Image(monthImg).select('total_precipitation')
        .multiply(1000) // m -> mm
        .clip(aoi)
        .rename('era5_precip');

      var precipDesc = 'era5_precip_' + y + '_' + mm;
      Export.image.toDrive({
        image: precip,
        description: precipDesc,
        folder: 'era5precip',
        fileNamePrefix: precipDesc,
        region: aoi,
        scale: 30, // resampled to match your Landsat 30m grid on export
        crs: 'EPSG:21097',
        maxPixels: 1e13
      });

      // ----- TEMPERATURE -----
      // band: mean_2m_air_temperature (Kelvin) - convert to Celsius
      var temp = ee.Image(monthImg).select('mean_2m_air_temperature')
        .subtract(273.15) // K -> °C
        .clip(aoi)
        .rename('era5_temp');

      var tempDesc = 'era5_temp_' + y + '_' + mm;
      Export.image.toDrive({
        image: temp,
        description: tempDesc,
        folder: 'era5temp',
        fileNamePrefix: tempDesc,
        region: aoi,
        scale: 30,
        crs: 'EPSG:21097',
        maxPixels: 1e13
      });
    })(year, month);
  }
}

print('ERA5 precip + temp export tasks queued: 1995-2025, monthly.');