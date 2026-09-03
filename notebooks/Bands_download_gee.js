// =========================================================================
// MULTI-TILE, MULTI-MISSION LANDSAT MOSAIC — STACKED 7-BAND EXPORT
// v6: float32 cast, explicit nodata bake-in, single combined priority score
//     (L5 > L7 whenever both present, era-correct otherwise), raster-mask
//     coverage diagnostic guarded against empty collections, chunked
//     getInfo() evaluation to avoid timeout, correct aoi.geometry() usage
// =========================================================================

var startYear = 1995;
var endYear = 2025;
var bandsOut = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'];

var l5l7Map = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6'];
var l8l9Map = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'];

// .area()/.intersection() require ee.Geometry, not Feature/FeatureCollection.
// filterBounds()/clip() accept aoi directly and don't need this.
var aoiGeometry = aoi.geometry ? aoi.geometry() : aoi;

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

// rank: lower number = higher intended priority. Combined into a single
// score with CLOUD_COVER so one sort() call is sufficient and unambiguous.
function prepImage(image, bandMap, missionTag, rank) {
  var masked = maskL2(image);
  var scaled = applyScaleFactors(masked);

  var cloudCover = ee.Number(image.get('CLOUD_COVER'));
  var priorityScore = ee.Number(rank).multiply(1000).add(cloudCover);

  return scaled.select(bandMap, bandsOut)
    .set('mission', missionTag)
    .set('sensor_rank', rank)
    .set('priority_score', priorityScore)
    .copyProperties(image, ['CLOUD_COVER', 'system:time_start']);
}

function getColl(collId, y, m, bandMap, missionTag, rank) {
  var startDate = ee.Date.fromYMD(y, m, 1);
  var endDate = startDate.advance(1, 'month');

  return ee.ImageCollection(collId)
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .map(function(img) { return prepImage(img, bandMap, missionTag, rank); });
}

// rank is passed explicitly at the call site per era — never a fixed
// property of the sensor itself.
function getL5(y, m, rank) { return getColl('LANDSAT/LT05/C02/T1_L2', y, m, l5l7Map, 'l5', rank); }
function getL7(y, m, rank) { return getColl('LANDSAT/LE07/C02/T1_L2', y, m, l5l7Map, 'l7', rank); }
function getL8(y, m, rank) { return getColl('LANDSAT/LC08/C02/T1_L2', y, m, l8l9Map, 'l8', rank); }
function getL9(y, m, rank) { return getColl('LANDSAT/LC09/C02/T1_L2', y, m, l8l9Map, 'l9', rank); }

// Era-based mission availability + priority. L5 always outranks L7 whenever
// both are queried in the same window (rank 1 vs rank 2).
function getMonthlyCollection(y, m) {
  var parts;
  var primaryTag;

  if (y <= 1998) {
    // L7 not yet launched (1999) — L5 only.
    parts = [getL5(y, m, 1)];
    primaryTag = 'l5';
  } else if (y <= 2012) {
    // Both L5 and L7 available. L5 prioritized per request.
    parts = [getL5(y, m, 1), getL7(y, m, 2)];
    primaryTag = 'l5';
  } else if (y <= 2021) {
    // L5 decommissioned (2013) — L8 primary, L7 fallback for sidelap/gap fill.
    parts = [getL8(y, m, 1), getL7(y, m, 2)];
    primaryTag = 'l8';
  } else {
    // L9 primary, L8 and L7 as fallbacks.
    parts = [getL9(y, m, 1), getL8(y, m, 2), getL7(y, m, 3)];
    primaryTag = 'l9';
  }

  var merged = parts[0];
  for (var i = 1; i < parts.length; i++) {
    merged = merged.merge(parts[i]);
  }

  return {coll: merged, primaryTag: primaryTag};
}

// Cheap raster-mask coverage diagnostic: fraction of AOI pixels that are
// valid (unmasked) in this month's mosaic, sampled at coarse scale via
// reduceRegion instead of exact vector polygon intersection. Guarded
// against empty collections, which otherwise produce a zero-band mosaic
// and crash .select() with "Invalid band number (0)".
function estimateCoverage(coll) {
  var hasImages = coll.size().gt(0);

  return ee.Number(
    ee.Algorithms.If(
      hasImages,
      coll.mosaic()
        .select(bandsOut)   // select by name, not position — safer
        .select(0)
        .mask()
        .reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: aoiGeometry,
          scale: 500,
          maxPixels: 1e9,
          bestEffort: true
        })
        .values()
        .get(0),
      -1   // sentinel: no scenes at all this month, coverage undefined
    )
  );
}

// BUILD FULL TASK LIST (client-side, cheap — no server evaluation yet)
var taskList = [];
for (var year = startYear; year <= endYear; year++) {
  for (var month = 1; month <= 12; month++) {
    var res = getMonthlyCollection(year, month);
    taskList.push({
      y: year,
      m: month,
      coll: res.coll,
      primaryTag: res.primaryTag,
      sizeQuery: res.coll.size(),
      coverageQuery: estimateCoverage(res.coll)
    });
  }
}

// EVALUATE SIZE + COVERAGE IN YEARLY CHUNKS (12 months at a time) instead
// of one 360-item getInfo() call, to avoid computation timeout / memory
// limit errors on the combined server-side graph.
var counts = [];
var coverageFractions = [];

for (var y2 = startYear; y2 <= endYear; y2++) {
  var yearTasks = taskList.filter(function(t) { return t.y === y2; });

  var yearSizeQueries = yearTasks.map(function(t) { return t.sizeQuery; });
  var yearCoverageQueries = yearTasks.map(function(t) { return t.coverageQuery; });

  var yearCounts = ee.List(yearSizeQueries).getInfo();
  var yearCoverage = ee.List(yearCoverageQueries).getInfo();

  counts = counts.concat(yearCounts);
  coverageFractions = coverageFractions.concat(yearCoverage);

  print('Evaluated year ' + y2 + ' (' + yearTasks.length + ' months)');
}

// EXPORT LOOP
taskList.forEach(function(entry, index) {
  var count = counts[index];
  var coverage = coverageFractions[index];

  if (count === 0) {
    print('Skipping ' + entry.y + '-' + entry.m + ': no scenes found.');
    return;
  }

  // Coverage is logged, not export-blocking — partial-coverage months are
  // retained on purpose; the RBFN gap-fills missing pixels using temporal
  // neighbors. -9999 nodata marks exactly which pixels those are.
  var coverageLabel = (coverage === null || coverage === -1) ? 'n/a' : coverage.toFixed(2);
  print('Exporting ' + entry.y + '-' + entry.m +
        ' (AOI coverage: ' + coverageLabel + ')');

  // Single combined-score sort. mosaic() gives priority last-to-first, so
  // the WORST score must sort first and the BEST (lowest) score last.
  var sorted = entry.coll.sort('priority_score', false);

  var composite = sorted.mosaic()
    .clip(aoi)
    .select(bandsOut)
    .unmask(-9999)   // bake explicit nodata into masked/off-footprint pixels
    .toFloat();       // halves file size vs float64, matches physical precision needed

  var mm = entry.m < 10 ? '0' + entry.m : '' + entry.m;
  var desc = 'landsat_' + entry.y + '_' + mm + '_' + entry.primaryTag;

  Export.image.toDrive({
    image: composite,
    description: desc,
    folder: 'Landsat_Bands',
    fileNamePrefix: desc,
    region: aoi,
    scale: 30,
    crs: 'EPSG:21097',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {
      cloudOptimized: true,
      noData: -9999
    }
  });
});

print('Queued monthly mosaic exports: float32, explicit nodata, ' +
      'L5>L7 priority, single combined mosaic score, ' +
      'coverage estimated via raster mask (empty-collection-safe) + chunked evaluation.');