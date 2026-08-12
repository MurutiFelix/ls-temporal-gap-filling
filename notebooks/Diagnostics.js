
// 1. CENTER MAP ON YOUR IMPORTED ASSET
Map.centerObject(aoi, 8); 
Map.addLayer(aoi, {color: 'red'}, 'Study Area'); 

var startYear = 1995; 
var endYear = 2010; 

// Load Surface Reflectance Tier 1 collections for Landsat 5 and 7 
var l5Col = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
  .filterBounds(aoi)
  .filter(ee.Filter.lt('CLOUD_COVER', 20));
var l7Col = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
  .filterBounds(aoi)
  .filter(ee.Filter.lt('CLOUD_COVER', 20));
var l8Col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') 
  .filterBounds(aoi) 
  .filter(ee.Filter.lt('CLOUD_COVER', 20)); 
var l9Col = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') 
  .filterBounds(aoi) 
  .filter(ee.Filter.lt('CLOUD_COVER', 20));

// Create a list of years and months to iterate over 
var years = ee.List.sequence(startYear, endYear); 
var months = ee.List.sequence(1, 12); 

// 3. NESTED ITERATION LOOP TO COUNT IMAGES PER MONTH 
var imageCounts = years.map(function(y) { 
  return months.map(function(m) { 
    // Set up start and end dates for the specific month 
    var startDate = ee.Date.fromYMD(y, m, 1); 
    var endDate = startDate.advance(1, 'month'); 
    
    // Filter collections by this specific month's date range 
    var l5Month = l5Col.filterDate(startDate, endDate); 
    var l7Month = l7Col.filterDate(startDate, endDate); 
    
    // Count the images 
    var l5Count = l5Month.size(); 
    var l7Count = l7Month.size(); 
    var totalCount = l5Count.add(l7Count); 
    
    // Return a feature with the counts as properties 
    return ee.Feature(null, { 
      'Year': y, 
      'Month': m, 
      'Landsat5_Count': l5Count, 
      'Landsat7_Count': l7Count, 
      'Total_Images': totalCount, 
      'Date_Label': ee.String(ee.Number(y).format('%d')).cat('-').cat(ee.String(ee.Number(m).format('%02d'))) 
    }); 
  }); 
}).flatten(); // Flatten the nested list into a single array 

// Convert the results into a FeatureCollection 
var countCollection = ee.FeatureCollection(imageCounts); 


// 4. DISPLAY IN CONSOLE AND EXPORT TO CSV 
print('Landsat Monthly Image Counts (1995-2025):', countCollection); 

// Export the results to your Google Drive as a clean CSV file 
Export.table.toDrive({ 
  collection: countCollection, 
  description: 'Landsat_Monthly_Image_Counts_1995_2025_20CV', 
  fileFormat: 'CSV', 
  selectors: ['Year', 'Month', 'Date_Label', 'Landsat5_Count', 'Landsat7_Count', 'Total_Images'] 
});
