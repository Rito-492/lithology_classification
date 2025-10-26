# Lithology Classification Data Visualization Report

## Data Overview

- **Total Samples**: 38,225
- **Features**: 4 (SP, GR, AC, DEPTH)
- **Wells**: 4 (Well IDs: 2, 146, 298, 2010)
- **Classes**: 3 lithology types (Class 0, 1, 2)

### Class Distribution
- Class 0: 9,989 samples (26.13%)
- Class 1: 4,764 samples (12.46%)
- Class 2: 23,472 samples (61.40%)

**Note**: Severe class imbalance issue - Class 2 accounts for over 60%

---

## Visualization Charts Description

### 1. 01_class_distribution.png - Lithology Class Distribution
**Content**:
- Left: Pie chart showing proportion of three lithology classes
- Right: Bar chart showing sample count for each class

**Key Findings**:
- Significant class imbalance exists
- Class 2 (61.4%) is the dominant lithology type
- Class 1 (12.5%) has fewer samples, may need oversampling or class weight adjustment

---

### 2. 02_feature_distributions.png - Feature Distribution Histograms
**Content**: Four subplots showing distributions of SP, GR, AC, DEPTH
- Red dashed line: Mean
- Green dashed line: Median

**Feature Statistics**:
- **SP**: Mean=54.98, Std=21.59, Range=[7.47, 476.40]
- **GR**: Mean=134.72, Std=34.21, Range=[10.00, 308.10]
- **AC**: Mean=293.89, Std=57.23, Range=[120.49, 567.06]
- **DEPTH**: Mean=1917.85, Std=688.16, Range=[301.04, 3316.88]

**Key Findings**:
- SP feature has extreme outliers (max=476.40 much larger than 75th percentile 61.22)
- Other features show relatively normal distributions
- Depth range spans widely from 301m to 3316m

---

### 3. 03_correlation_matrix.png - Feature Correlation Heatmap
**Content**: Shows Pearson correlation coefficients between four features

**Key Findings**:
- Identifies linear relationships between features
- Feature pairs with high correlation (|r| > 0.7) may be redundant
- Useful for feature selection and multicollinearity diagnosis

---

### 4. 04_boxplots_by_class.png - Feature Boxplots by Class
**Content**: Shows feature distribution for each lithology class
- Box: Interquartile range (IQR)
- Horizontal line: Median
- Diamond: Mean
- Whiskers: 1.5×IQR range
- Circles: Outliers

**Use Cases**:
- Compare differences in log responses across lithology types
- Identify which features have best discriminative power for classification
- Discover outliers and anomalies

---

### 5. 05_scatter_matrix.png - Feature Scatter Matrix
**Content**: Pairwise scatter plots of three log features (SP, GR, AC)
- Diagonal: Histogram distributions for each class
- Off-diagonal: Scatter plots of feature pairs, colored by class
- Sampled 5000 points for faster rendering

**Use Cases**:
- Observe relationship patterns between features
- Identify class distributions in feature space
- Discover potential clustering patterns

---

### 6. 06_well_logs.png - Well Log Curves
**Content**: Well log display for a single well (well with most samples)
- First three columns: SP, GR, AC log curves
- Fourth column: Lithology column (different colors for different lithologies)
- Depth range: Displays 200 meters continuous interval

**Use Cases**:
- Observe log response trends with depth
- Understand relationship between lithology changes and log responses
- Identify lithological boundaries and formation characteristics

---

### 7. 07_well_statistics.png - Well Statistics
**Content**: Four subplots showing statistical characteristics of each well
- Top left: Sample count by well
- Top right: Depth range by well
- Bottom left: Lithology class distribution by well (stacked bar chart)
- Bottom right: Lithology class percentage by well

**Well Statistics**:
- Well 2: 16,776 samples
- Well 146: 14,193 samples
- Well 298: 5,751 samples
- Well 2010: 1,505 samples

**Key Findings**:
- Large variation in sample counts between wells
- Can observe consistency of lithology distribution across wells
- Useful for assessing data representativeness

---

### 8. 08_feature_by_depth.png - Features vs Depth
**Content**: Three scatter plots showing SP, GR, AC variation with depth
- Different colors represent different lithology classes
- Sampled 10,000 points

**Use Cases**:
- Observe log parameter trends with burial depth
- Identify depth-related lithology distribution patterns
- Discover geological phenomena like compaction and diagenesis

---

## Data Quality Assessment

### Strengths
✓ Sufficient data volume (38,225 records)
✓ Complete features with no obvious missing values
✓ Multiple well data providing representativeness

### Issues to Note
⚠ Severe class imbalance (61% vs 12%)
⚠ SP feature contains outliers
⚠ Large variation in sample counts between wells

### Recommendations

1. **Class Imbalance Handling**: 
   - Use class weights (class_weight parameter)
   - SMOTE or other oversampling methods
   - Stratified sampling to ensure balanced validation set

2. **Outlier Treatment**:
   - Detect and handle outliers in SP feature
   - Consider robust normalization methods (e.g., RobustScaler)

3. **Feature Engineering**:
   - Feature selection based on correlation analysis
   - Consider creating feature interaction terms
   - Try log transformation or standardization

4. **Model Validation**:
   - Use stratified K-fold cross-validation
   - Focus on confusion matrix, especially minority class recognition rate
   - Use balanced metrics like F1-score or AUC-ROC

---

## Running Visualization

To regenerate these charts, run:
```bash
cd lithology_classification
python visualize.py
```

All charts will be saved in the `visualizations/` directory.

---

## Chart Improvements

✓ **Fixed encoding issues** - All labels now in English
✓ **High resolution** - Charts saved at 150 DPI for clarity
✓ **Professional styling** - Clean, publication-ready appearance
✓ **Clear legends** - Easy to interpret class and feature information
