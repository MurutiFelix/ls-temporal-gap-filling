
// =========================================================================
// MULTI-TILE, MULTI-MISSION LANDSAT MOSAIC — STACKED 7-BAND EXPORT
// Single batched size check 
// =========================================================================

// CONFIG
var startYear = 1995;
var endYear = 2025;
var bandsOut = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'];

var l5l7Map = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6'];
var l8l9Map = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'];

// CLOUD, SHADOW, AND SATURATION MASKING
function maskL2(image) {
  var qa = image.select('QA_PIXEL');
  var cloudShadowBitMask = (1 << 3) | (1 << 4);
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0);
  var satMask = image.select('QA_RADSAT').eq(0);
  return image.updateMask(mask).updateMask(satMask);
}

function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

function prepImage(image, bandMap, rank, missionTag) {
  var masked = maskL2(image);
  var scaled = applyScaleFactors(masked);
  return scaled.select(bandMap, bandsOut)
    .set('sensor_rank', rank)
    .set('mission', missionTag)
    .copyProperties(image, ['CLOUD_COVER', 'system:time_start']);
}

// COLLECTION GETTERS (Strict Calendar Month)
function getColl(collId, y, m, bandMap, rank, missionTag) {
  var startDate = ee.Date.fromYMD(y, m, 1);
  var endDate = startDate.advance(1, 'month');

  return ee.ImageCollection(collId)
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .map(function(img) { return prepImage(img, bandMap, rank, missionTag); });
}

function getL5(y, m) { return getColl('LANDSAT/LT05/C02/T1_L2', y, m, l5l7Map, 2, 'l5'); }
function getL7(y, m) { return getColl('LANDSAT/LE07/C02/T1_L2', y, m, l5l7Map, 1, 'l7'); }
function getL8(y, m) { return getColl('LANDSAT/LC08/C02/T1_L2', y, m, l8l9Map, 3, 'l8'); }
function getL9(y, m) { return getColl('LANDSAT/LC09/C02/T1_L2', y, m, l8l9Map, 3, 'l9'); }

function getMonthlyCollection(y, m) {
  var collections;
  var primaryTag;

  if (y <= 1998) {
    collections = [getL5(y, m)];
    primaryTag = 'l5';
  } else if (y <= 2003) {
    collections = [getL7(y, m), getL5(y, m)];
    primaryTag = 'l7';
  } else if (y <= 2012) {
    collections = [getL5(y, m), getL7(y, m)];
    primaryTag = 'l5';
  } else if (y <= 2021) {
    collections = [getL8(y, m), getL7(y, m)];
    primaryTag = 'l8';
  } else {
    collections = [getL9(y, m), getL8(y, m), getL7(y, m)];
    primaryTag = 'l9';
  }

  var merged = ee.ImageCollection(collections[0]);
  for (var i = 1; i < collections.length; i++) {
    merged = merged.merge(collections[i]);
  }
  return {coll: merged, primaryTag: primaryTag};
}

// PRE-COMPUTE ALL COLLECTION SIZES IN ONE SERVER ROUND-TRIP
var taskList = [];
var sizeQueries = [];
for (var year = startYear; year <= endYear; year++) {
  for (var month = 1; month <= 12; month++) {
    var res = getMonthlyCollection(year, month);
    taskList.push({y: year, m: month, coll: res.coll, primaryTag: res.primaryTag});
    sizeQueries.push(res.coll.size());
  }
}
var counts = ee.List(sizeQueries).getInfo(); // single blocking call for the whole run

// EXPORT LOOP — pure client-side after this point, no more server round-trips
taskList.forEach(function(entry, index) {
  var count = counts[index];
  if (count === 0) {
    print('Skipping ' + entry.y + '-' + entry.m + ': no scenes found.');
    return;
  }

  var sorted = entry.coll.sort('CLOUD_COVER', false).sort('sensor_rank', true);
  var composite = sorted.mosaic().clip(aoi);

  var mm = entry.m < 10 ? '0' + entry.m : '' + entry.m;
  var desc = 'landsat_' + entry.y + '_' + mm + '_' + entry.primaryTag;

  Export.image.toDrive({
    image: composite.select(bandsOut),
    description: desc,
    folder: 'Landsat_Bands',
    fileNamePrefix: desc,
    region: aoi,
    scale: 30,
    crs: 'EPSG:21097',
    maxPixels: 1e13
  });
});
print('Successfully queued monthly mosaic tasks!');