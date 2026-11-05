# Command-Line Interface Guide

## Overview

The EaglePlot CLI provides a powerful command-line interface for scientific data visualization and analysis. This guide covers all commands, options, and usage examples.

## Installation

The CLI is included in the EaglePlot package. No additional installation is required.

## Basic Usage

```bash
eagleplot [OPTIONS] COMMAND [ARGS...]
```

## Global Options

- `--help`, `-h`: Show help message and exit
- `--version`: Show version and exit
- `--verbose`, `-v`: Enable verbose output (detailed logging)
- `--quiet`, `-q`: Suppress informational messages (errors only)
- `--examples`: Show usage examples and exit

## Commands

### 1. plot - Create Plots

Create various types of plots from data files.

**Syntax:**
```bash
eagleplot plot --input FILE --type TYPE [OPTIONS]
```

**Required Arguments:**
- `--input`, `-i FILE`: Input data file path
- `--type`, `-t TYPE`: Type of plot to create

**Plot Types:**
- `histogram`: Histogram plot
- `line`: Line plot
- `scatter`: Scatter plot
- `heatmap`: 2D heatmap
- `contour`: Contour plot
- `distribution`: Distribution plot (Seaborn, if available)
- `box`: Box/violin plot (Seaborn, if available)
- `regression`: Regression plot (Seaborn, if available)
- `interactive_histogram`: Interactive histogram (Plotly, if available)
- `interactive_scatter`: Interactive scatter (Plotly, if available)
- `interactive_line`: Interactive line plot (Plotly, if available)

**Optional Arguments:**
- `--output`, `-o FILE`: Output file path (default: plot.png or plot.html)
- `--column NAME`: Column name to extract from DataFrame
- `--title TEXT`: Plot title
- `--xlabel TEXT`: X-axis label
- `--ylabel TEXT`: Y-axis label
- `--color COLOR`: Line/marker color
- `--style STYLE`: Line style (-, --, -., :)
- `--marker MARKER`: Marker style (o, s, ^, etc.)
- `--xlim MIN,MAX`: X-axis limits (e.g., "0,10")
- `--ylim MIN,MAX`: Y-axis limits (e.g., "0,100")
- `--bins N`: Number of bins for histogram (default: 30)
- `--dpi N`: Output resolution in DPI (default: 300)
- `--dist-kind KIND`: Distribution kind: hist, kde, ecdf
- `--x-column NAME`: X column for box/scatter plots
- `--y-column NAME`: Y column for box/scatter plots
- `--order N`: Polynomial order for regression (default: 1)

**Examples:**

```bash
# Simple histogram
eagleplot plot --input data.npy --type histogram --output hist.png

# Customized histogram
eagleplot plot --input data.csv --column Temperature --type histogram \
    --bins 50 --title "Temperature Distribution" \
    --xlabel "Temperature (K)" --ylabel "Frequency" \
    --output temp_dist.png

# Line plot with styling
eagleplot plot --input timeseries.csv --type line \
    --title "Time Series Data" --color blue --style "--" \
    --marker o --output timeseries.png

# Interactive plot
eagleplot plot --input data.npy --type interactive_histogram \
    --output interactive.html --bins 40

# Scatter plot with limits
eagleplot plot --input points.csv --type scatter \
    --xlim "0,10" --ylim "0,100" --output scatter.png
```

### 2. stats - Calculate Statistics

Calculate and display statistical summary of data.

**Syntax:**
```bash
eagleplot stats --input FILE [OPTIONS]
```

**Required Arguments:**
- `--input`, `-i FILE`: Input data file path

**Optional Arguments:**
- `--output`, `-o FILE`: Output file for statistics (JSON format)
- `--column NAME`: Column name to analyze from DataFrame
- `--normality`: Perform Shapiro-Wilk normality test

**Output Statistics:**
- mean, median, std, var
- min, max, range
- q1 (25th percentile), q3 (75th percentile), iqr
- Normality test results (if --normality flag used)

**Examples:**

```bash
# Basic statistics
eagleplot stats --input data.npy

# Statistics for specific column
eagleplot stats --input data.csv --column Temperature

# Statistics with normality test
eagleplot stats --input data.csv --column Values --normality

# Save statistics to file
eagleplot stats --input data.npy --output stats.json --normality
```

### 3. batch - Batch Processing

Process multiple files using a configuration file.

**Syntax:**
```bash
eagleplot batch --config FILE
```

**Required Arguments:**
- `--config`, `-c FILE`: Configuration file (JSON or YAML)

**Configuration Format (JSON):**

```json
{
  "tasks": [
    {
      "name": "Task 1 Description",
      "command": "plot",
      "input": "data1.npy",
      "type": "histogram",
      "output": "output1.png",
      "bins": 50,
      "title": "Histogram 1"
    },
    {
      "name": "Task 2 Description",
      "command": "stats",
      "input": "data2.csv",
      "column": "Temperature",
      "output": "stats.json",
      "normality": true
    }
  ]
}
```

**Configuration Format (YAML):**

```yaml
tasks:
  - name: Task 1 Description
    command: plot
    input: data1.npy
    type: histogram
    output: output1.png
    bins: 50
    title: Histogram 1

  - name: Task 2 Description
    command: stats
    input: data2.csv
    column: Temperature
    output: stats.json
    normality: true
```

**Examples:**

```bash
# Run batch processing
eagleplot batch --config batch_config.json

# Run with verbose output
eagleplot --verbose batch --config analysis_pipeline.yaml
```

### 4. compare - Compare Datasets

Compare multiple datasets on the same plot.

**Syntax:**
```bash
eagleplot compare --inputs FILE1 FILE2 [FILE3...] [OPTIONS]
```

**Required Arguments:**
- `--inputs FILE1 FILE2 ...`: Input data files to compare

**Optional Arguments:**
- `--output`, `-o FILE`: Output plot file (default: comparison.png)
- `--type`, `-t TYPE`: Comparison plot type (histogram, line, scatter)
- `--labels LABEL1 LABEL2 ...`: Labels for each dataset
- `--column NAME`: Column name to extract from DataFrames
- `--title TEXT`: Plot title
- `--xlabel TEXT`: X-axis label
- `--ylabel TEXT`: Y-axis label
- `--bins N`: Number of bins for histogram
- `--dpi N`: Output resolution in DPI

**Examples:**

```bash
# Compare with histogram
eagleplot compare --inputs data1.csv data2.csv data3.csv \
    --type histogram --output comparison.png

# Compare with labels
eagleplot compare --inputs trial1.npy trial2.npy trial3.npy \
    --type line --labels "Trial 1" "Trial 2" "Trial 3" \
    --title "Trial Comparison" --output trials.png

# Compare specific column
eagleplot compare --inputs exp1.csv exp2.csv \
    --column Temperature --type histogram --bins 40 \
    --output temp_comparison.png
```

### 5. convert - Convert Data Formats

Convert data between different file formats.

**Syntax:**
```bash
eagleplot convert --input FILE --output FILE
```

**Required Arguments:**
- `--input`, `-i FILE`: Input data file
- `--output`, `-o FILE`: Output data file (format auto-detected from extension)

**Supported Formats:**
- `.pkl` - Pickle (Python objects)
- `.npy` - NumPy array (single)
- `.npz` - NumPy arrays (multiple, compressed)
- `.csv` - CSV (comma-separated values)
- `.xlsx`, `.xls` - Excel spreadsheet
- `.tsv` - TSV (tab-separated values)
- `.json` - JSON format
- `.parquet` - Parquet (columnar storage)

**Examples:**

```bash
# Convert pickle to CSV
eagleplot convert --input data.pkl --output data.csv

# Convert NumPy to Excel
eagleplot convert --input arrays.npy --output data.xlsx

# Convert CSV to Parquet
eagleplot convert --input large_data.csv --output large_data.parquet
```

## Advanced Usage

### Verbose Mode

Use `--verbose` before the command for detailed logging:

```bash
eagleplot --verbose plot --input data.npy --type histogram --output plot.png
```

Output:
```
[INFO] Loading data from data.npy
[INFO] Creating histogram plot
[INFO] Plot saved to plot.png
```

### Quiet Mode

Use `--quiet` to suppress all non-error messages:

```bash
eagleplot --quiet batch --config large_pipeline.json
```

### Chaining with Shell Commands

Use CLI in shell pipelines:

```bash
# Find all .npy files and create histograms
for file in examples/data/*.npy; do
    eagleplot plot --input "$file" --type histogram \
        --output "plots/$(basename "$file" .npy)_hist.png"
done

# Process with logging
eagleplot --verbose stats --input data.csv 2>&1 | tee analysis.log
```

### Integration with Scripts

Use the CLI from shell scripts:

```bash
#!/bin/bash
# analysis_pipeline.sh

echo "Running analysis pipeline..."

# Step 1: Generate plots
eagleplot plot --input raw_data.csv --type line --output plot1.png
eagleplot plot --input raw_data.csv --type histogram --output plot2.png

# Step 2: Calculate statistics
eagleplot stats --input raw_data.csv --output stats.json --normality

# Step 3: Compare with reference
eagleplot compare --inputs raw_data.csv reference.csv \
    --type line --labels "Current" "Reference" --output comparison.png

echo "Analysis complete!"
```

## Error Handling

The CLI provides clear error messages:

**File Not Found:**
```
[ERROR] File not found: File not found: nonexistent.npy
```

**Invalid Arguments:**
```
eagleplot/cli.py: error: argument --type: invalid choice: 'invalid_type'
```

**Missing Required Arguments:**
```
eagleplot/cli.py: error: the following arguments are required: --input
```

**Data Format Errors:**
```
[ERROR] Invalid value: Scatter plot requires 2D data (x, y)
```

## Tips and Best Practices

1. **Use Batch Processing**: For multiple operations, use batch mode instead of running individual commands

2. **Specify Output Paths**: Always use `--output` to control where files are saved

3. **Check Data First**: Use `stats` command to understand your data before plotting

4. **Use Verbose Mode for Debugging**: If something goes wrong, run with `--verbose` to see detailed logs

5. **Leverage Compare Command**: Compare multiple datasets in a single plot instead of creating separate plots

6. **Convert Formats**: Use `convert` to prepare data in the format needed for other tools

7. **Save Statistics**: Always save statistics to JSON files for later reference and record-keeping

8. **Use Configuration Files**: Store batch configurations for repeatable analyses

## Keyboard Shortcuts

When running interactively:
- `Ctrl+C`: Interrupt current operation
- `Ctrl+D` / `Ctrl+Z`: Exit (on Unix/Windows respectively)

## Exit Codes

- `0`: Success
- `1`: Error (file not found, invalid data, etc.)
- `2`: Invalid arguments

## Getting Help

```bash
# General help
eagleplot --help

# Command-specific help
eagleplot plot --help
eagleplot stats --help
eagleplot batch --help
eagleplot compare --help
eagleplot convert --help

# Show examples
eagleplot --examples
```

## Troubleshooting

**Problem**: "Command not found: python"
- **Solution**: Use `python3` instead, or ensure Python is in your PATH

**Problem**: "Module not found: modules.io"
- **Solution**: Run from the eagleplot directory, not the examples/ directory

**Problem**: Plot files are too large
- **Solution**: Reduce `--dpi` value (e.g., `--dpi 150` instead of default 300)

**Problem**: Interactive plots don't work
- **Solution**: Ensure Plotly is installed: `pip install plotly`

**Problem**: Distribution/box plots not available
- **Solution**: Ensure Seaborn is installed: `pip install seaborn`

## See Also

- Python API documentation in `modules/`
- Example scripts in `examples/`
- Test suite in `tests/test_cli.py`
- Project roadmap in `roadmap.md`

---

**Version**: 1.0.0
**Last Updated**: 2026-02-12
**Project**: EaglePlot
