// CONFIG 
var startYear = 1995;
var endYear = 2025;
var cloudThresh = 20;
var bandsOut = ['green', 'blue', 'red', 'nir', 'swir1', 'swir2', 'thermal'];
var l5l7Map = ['SR_B2', 'SR_B1', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6'];
var l8l9Map = ['SR_B3', 'SR_B2', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'];

// CLOUD MASKING 
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
function prepImage(image, bandMap) {
  var masked = maskL2(image);
  var scaled = applyScaleFactors(masked);
  return scaled.select(bandMap, bandsOut);
}

// COLLECTION GETTERS 
function getColl(collId, y, m, bandMap) {
  return ee.ImageCollection(collId)
    .filterBounds(aoi)
    .filterDate(ee.Date.fromYMD(y, m, 1), ee.Date.fromYMD(y, m, 1).advance(1, 'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', cloudThresh))
    .map(function(img) { return prepImage(img, bandMap); });
}
function getL5(y, m) { return getColl('LANDSAT/LT05/C02/T1_L2', y, m, l5l7Map); }
function getL7(y, m) { return getColl('LANDSAT/LE07/C02/T1_L2', y, m, l5l7Map); }
function getL8(y, m) { return getColl('LANDSAT/LC08/C02/T1_L2', y, m, l8l9Map); }
function getL9(y, m) { return getColl('LANDSAT/LC09/C02/T1_L2', y, m, l8l9Map); }

function getMissionColl(name, y, m) {
  if (name === 'l5') return getL5(y, m);
  if (name === 'l7') return getL7(y, m);
  if (name === 'l8') return getL8(y, m);
  return getL9(y, m);
}

// MISSION PRIORITY LIST BY ERA 
// Ordered candidates, tried in sequence until one has data.
function getMissionPriority(y) {
  if (y <= 1998) return ['l5', 'l7'];
  if (y <= 2012) return ['l7', 'l5'];
  if (y === 2013) return ['l8', 'l7'];       // transition year
  if (y <= 2021) return ['l8', 'l7'];        // L9 doesn't exist yet
  return ['l9', 'l8', 'l7'];                 // 2022+: L9 primary, L8 fallback
}

// BUILD FULL TASK LIST SERVER-SIDE, RESOLVE ONCE 
var monthKeys = [];
var sizeQueries = [];

for (var year = startYear; year <= endYear; year++) {
  for (var month = 1; month <= 12; month++) {
    var priority = getMissionPriority(year);
    var colls = priority.map(function(name) { return getMissionColl(name, year, month); });

    monthKeys.push({y: year, m: month, priority: priority});
    colls.forEach(function(c) { sizeQueries.push(c.size()); });
  }
}

// ONE round-trip for the whole run instead of one per month
var allSizes = ee.List(sizeQueries).getInfo();

//  MAIN EXPORT LOOP 
var sizeIdx = 0;
monthKeys.forEach(function(entry) {
  var priority = entry.priority;
  var sizesForMonth = priority.map(function() { return allSizes[sizeIdx++]; });

  // pick first mission in priority order that actually has data
  var missionUsed = null;
  for (var i = 0; i < priority.length; i++) {
    if (sizesForMonth[i] > 0) { missionUsed = priority[i]; break; }
  }

  if (!missionUsed) {
    print('No data for', entry.y, entry.m, '- skipping');
    return;
  }

  var coll = getMissionColl(missionUsed, entry.y, entry.m);
  var composite = coll.median().clip(aoi);
  var mm = entry.m < 10 ? '0' + entry.m : '' + entry.m;

  bandsOut.forEach(function(bandName) {
    var img = composite.select([bandName]);
    var desc = bandName + '_' + entry.y + '_' + mm + '_' + missionUsed;

    Export.image.toDrive({
      image: img,
      description: desc,
      folder: 'Landsat_Bands',
      fileNamePrefix: desc,
      region: aoi,
      scale: 30,
      crs: 'EPSG:21097',
      maxPixels: 1e13
    });
  });
});