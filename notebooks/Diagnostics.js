
// 1. CENTER MAP ON YOUR IMPORTED ASSET

Map.centerObject(aoi, 8);
Map.addLayer(aoi, {color: 'red'}, 'Study Area');

var startYear = 1995;
var endYear = 2025;
var cloudThresh = 20;


// 2. LOAD ALL MISSIONS, BOTH LEVEL-2 (SR/ST) AND LEVEL-1 (TOA/radiance)

// Level-2: science-ready, atmospherically corrected 
var l2Collections = {
  Landsat5: ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', cloudThresh)),
  Landsat7: ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', cloudThresh)),
  Landsat8: ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', cloudThresh)),
  Landsat9: ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', cloudThresh))
};

// Level-1: geometrically corrected only 
var l1Collections = {
  Landsat5: ee.ImageCollection('LANDSAT/LT05/C02/T1_TOA').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', cloudThresh)),
  Landsat7: ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', cloudThresh)),
  Landsat8: ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', cloudThresh)),
  Landsat9: ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA').filterBounds(aoi).filter(ee.Filter.lt('CLOUD_COVER', cloudThresh))
};

var missionNames = Object.keys(l2Collections); // same 4 names for both dicts


// 3. NESTED ITERATION LOOP - COUNT PER MISSION, PER LEVEL, PER MONTH

var years = ee.List.sequence(startYear, endYear);
var months = ee.List.sequence(1, 12);

var imageCounts = years.map(function(y) {
  return months.map(function(m) {
    var startDate = ee.Date.fromYMD(y, m, 1);
    var endDate = startDate.advance(1, 'month');

    var props = {
      'Year': y,
      'Month': m,
      'Date_Label': ee.String(ee.Number(y).format('%d')).cat('-').cat(ee.String(ee.Number(m).format('%02d')))
    };

    var totalL2 = ee.Number(0);
    var totalL1 = ee.Number(0);

    missionNames.forEach(function(name) {
      var l2Count = l2Collections[name].filterDate(startDate, endDate).size();
      var l1Count = l1Collections[name].filterDate(startDate, endDate).size();

      props[name + '_L2_Count'] = l2Count;
      props[name + '_L1_Count'] = l1Count;

      totalL2 = totalL2.add(l2Count);
      totalL1 = totalL1.add(l1Count);
    });

    props['Total_L2_Images'] = totalL2;
    props['Total_L1_Images'] = totalL1;
    // Positive gap = scenes exist at L1 but didn't make it into L2 SR/ST
    props['L1_minus_L2_Gap'] = totalL1.subtract(totalL2);

    return ee.Feature(null, props);
  });
}).flatten();

var countCollection = ee.FeatureCollection(imageCounts);


// 4. DISPLAY IN CONSOLE AND EXPORT TO CSV

print('Landsat Monthly Image Counts, all missions, L1 vs L2 (1995-2025):', countCollection);

Export.table.toDrive({
  collection: countCollection,
  description: 'Landsat_Monthly_Image_Counts_1995_2025_AllMissions_L1_L2_20CV',
  fileFormat: 'CSV',
  selectors: [
    'Year', 'Month', 'Date_Label',
    'Landsat5_L2_Count', 'Landsat7_L2_Count', 'Landsat8_L2_Count', 'Landsat9_L2_Count', 'Total_L2_Images',
    'Landsat5_L1_Count', 'Landsat7_L1_Count', 'Landsat8_L1_Count', 'Landsat9_L1_Count', 'Total_L1_Images',
    'L1_minus_L2_Gap'
  ]
});