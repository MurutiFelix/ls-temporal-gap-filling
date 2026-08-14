
// AVHRR NDVI DOWNLOAD - GAP MONTHS ONLY (validation reference for RBFN fill)
// Requires: aoi already imported as an asset in this script.

// ----- 1. LIST YOUR KNOWN GAP MONTHS -----
var gapMonths = [
  {y: 1995, m: 6},
  {y: 1995, m: 9},
  {y: 1995, m: 10},
  {y: 1995, m: 11},
  {y: 1995, m: 12},
  {y: 1996, m: 1},
  {y: 1996, m: 2},
  {y: 1996, m: 3},
  {y: 1996, m: 4},
  {y: 1996, m: 5},
  {y: 1996, m: 6},
  {y: 1996, m: 7},
  {y: 1996, m: 9},
  {y: 1996, m: 11},
  {y: 1997, m: 1},
  {y: 1997, m: 2},
  {y: 1997, m: 3},
  {y: 1997, m: 4},
  {y: 1997, m: 6},
  {y: 1997, m: 8},
  {y: 1997, m: 9},
  {y: 1997, m: 11},
  {y: 1997, m: 12},
  {y: 1998, m: 1},
  {y: 1998, m: 2},
  {y: 1998, m: 3},
  {y: 1998, m: 4},
  {y: 1998, m: 5},
  {y: 1998, m: 7},
  {y: 1998, m: 8},
  {y: 1998, m: 10},
  {y: 1998, m: 11},
  {y: 1998, m: 12},
  {y: 1999, m: 1},
  {y: 1999, m: 2},
  {y: 1999, m: 3},
  {y: 1999, m: 6},
  {y: 1999, m: 7},
  {y: 1999, m: 11},
  {y: 1999, m: 12},
  {y: 2000, m: 11},
  {y: 2001, m: 1},
  {y: 2001, m: 11},
  {y: 2001, m: 12},
  {y: 2003, m: 6},
  {y: 2003, m: 9},
  {y: 2003, m: 11},
  {y: 2005, m: 4},
  {y: 2006, m: 3},
  {y: 2006, m: 12},
  {y: 2007, m: 6},
  {y: 2007, m: 11},
  {y: 2008, m: 7},
  {y: 2009, m: 5},
  {y: 2010, m: 6},
  {y: 2010, m: 11},
  {y: 2011, m: 5},
  {y: 2011, m: 6},
  {y: 2011, m: 11},
  {y: 2012, m: 8},
  {y: 2016, m: 4},
  {y: 2019, m: 12},
  {y: 2024, m: 7}
];
// ----- 2. LOAD AVHRR NDVI COLLECTION -----
var avhrrCol = ee.ImageCollection('NOAA/CDR/AVHRR/NDVI/V5')
  .filterBounds(aoi)
  .select('NDVI'); // scaled NDVI band; SDS_NDVI has a scale factor of 0.0001 - handled below

// ----- 3. EXPORT LOOP - ONE COMPOSITE PER GAP MONTH -----
gapMonths.forEach(function(entry) {
  var y = entry.y;
  var m = entry.m;

  var startDate = ee.Date.fromYMD(y, m, 1);
  var endDate = startDate.advance(1, 'month');

  var monthColl = avhrrCol.filterDate(startDate, endDate);

  // AVHRR NDVI CDR is already QA-masked upstream, but you can apply
  // an additional per-pixel mask via the QA band if you want stricter filtering:
  // var qaCol = ee.ImageCollection('NOAA/CDR/AVHRR/NDVI/V5').select('QA');

  var composite = monthColl.median()
    .multiply(0.0001) // apply CDR scale factor to get true NDVI (-1 to 1)
    .clip(aoi)
    .rename('avhrr_ndvi');

  var mm = m < 10 ? '0' + m : '' + m;
  var desc = 'avhrr_ndvi_' + y + '_' + mm;

  Export.image.toDrive({
    image: composite,
    description: desc,
    folder: 'Datasets1_AVHRR_Validation',
    fileNamePrefix: desc,
    region: aoi,
    scale: 5000, // native AVHRR CDR resolution (~5km)
    crs: 'EPSG:21097',
    maxPixels: 1e13
  });
});

print('AVHRR export tasks queued for', gapMonths.length, 'gap months.');