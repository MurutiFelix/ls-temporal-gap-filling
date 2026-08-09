// ===== CONFIG =====
var startYear = 1995;
var endYear = 2005;
var cloudThresh = 20;

var bandsOut = ['green', 'blue', 'red', 'nir', 'swir1', 'swir2', 'thermal'];

var l5l7Map = ['SR_B2', 'SR_B1', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6'];
var l8l9Map = ['SR_B3', 'SR_B2', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'];

// ===== CLOUD MASKING (Collection 2 SR/ST QA_PIXEL) =====
function maskL2(image) {
  var qa = image.select('QA_PIXEL');
  var cloudShadowBitMask = (1 << 3) | (1 << 4); // cloud, cloud shadow
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0);
  var satMask = image.select('QA_RADSAT').eq(0);
  return image.updateMask(mask).updateMask(satMask);
}

// Collection 2 Level-2 scale factors
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

function prepImage(image, bandMap) {
  var masked = maskL2(image);
  var scaled = applyScaleFactors(masked);
  return scaled.select(bandMap, bandsOut);
}

// ===== COLLECTION GETTERS =====
function getL5(y, m) {
  return ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(ee.Date.fromYMD(y, m, 1), ee.Date.fromYMD(y, m, 1).advance(1, 'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', cloudThresh))
    .map(function(img) { return prepImage(img, l5l7Map); });
}

function getL7(y, m) {
  return ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(ee.Date.fromYMD(y, m, 1), ee.Date.fromYMD(y, m, 1).advance(1, 'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', cloudThresh))
    .map(function(img) { return prepImage(img, l5l7Map); });
}

function getL8(y, m) {
  return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(ee.Date.fromYMD(y, m, 1), ee.Date.fromYMD(y, m, 1).advance(1, 'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', cloudThresh))
    .map(function(img) { return prepImage(img, l8l9Map); });
}

// ===== MISSION SELECTION WITH FAILSAFES =====
// 1995-98: L5 primary, L7 fallback
// 1999-2005: L7 primary, L5 fallback (L8 doesn't exist yet pre-2013, so no need to reach for it here)
function getMonthlyCollection(y, m) {
  var primary, fallback;
  if (y <= 1998) {
    primary = getL5(y, m);
    fallback = getL7(y, m);
  } else {
    primary = getL7(y, m);
    fallback = getL5(y, m);
  }
  return ee.ImageCollection(primary.merge(fallback));
}

// ===== MAIN EXPORT LOOP =====
for (var year = startYear; year <= endYear; year++) {
  for (var month = 1; month <= 12; month++) {
    (function(y, m) {
      var monthColl = getMonthlyCollection(y, m);
      var count = monthColl.size();

      var composite = ee.Image(ee.Algorithms.If(
        count.gt(0),
        monthColl.median().clip(aoi),
        null
      ));

      var mm = m < 10 ? '0' + m : '' + m;

      bandsOut.forEach(function(bandName) {
        var img = ee.Image(composite).select([bandName]);
        var desc = bandName + '_' + y + '_' + mm;

        Export.image.toDrive({
          image: img,
          description: desc,
          folder: 'Datasets1',
          fileNamePrefix: desc,
          region: aoi,
          scale: 30,
          crs: 'EPSG:21097',
          maxPixels: 1e13
        });
      });
    })(year, month);
  }
}