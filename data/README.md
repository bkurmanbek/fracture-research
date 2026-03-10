# Data Files

## Files

### `Teapod_fractures_XY_coordinates.csv`

Raw XY coordinates of mapped fracture traces from the Teapot Dome field (Naval Petroleum Reserve No. 3, Wyoming, USA). Each row is one point along a fracture trace identified by a fracture ID column.

### `train_fractures_processed.csv`

Preprocessed training set. Contains **1,197 fractures** and **5,541 points** after geographic exclusion of the southeastern test region.

### `test_fractures_processed.csv`

Preprocessed test set. Contains **139 fractures** and **625 points**. Geographic bounds of the test region: x ∈ [795,000–805,000] ft, y ∈ [950,000–970,000] ft (~18.6 km²).

---

## Column Reference

| Column | Description |
|--------|-------------|
| `fracture_id` | Integer identifier for each fracture trace |
| `point_idx` | Zero-based index of the point along its fracture |
| `coord_x` | UTM Easting coordinate (feet) |
| `coord_y` | UTM Northing coordinate (feet) |
| `prev_angle` | Bearing of the incoming segment (radians) |
| `next_angle` | Bearing of the outgoing segment (radians) |
| `prev_length` | Length of the incoming segment (feet) |
| `next_length` | Length of the outgoing segment (feet) |
| `curvature` | Local curvature κ = |Δangle| / segment_length |
| `closest_seg_{0-3}_p{1,2}_{x,y}` | XY endpoints of the 4 nearest neighbour fracture segments (8 columns × 4 neighbours = 16 columns) |

Total columns: 25

---

## Notes

- **28 fractures** in the test set have only 2 points. Cases 2–4 exclude these because their preprocessors require ≥ 3 points to compute `prev_angle`, `next_angle`, and `curvature`. Case 1 evaluates all 139 fractures.
- Coordinates are in US survey feet (EPSG:4267 / Wyoming State Plane or similar). Do not confuse with metres — segment lengths are on the order of hundreds of feet.
- The train/test split is **geographic**, not random. The test region is spatially separated from the training area to simulate true out-of-sample prediction.
