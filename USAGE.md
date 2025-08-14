# 🚀 Keras-TensorFlow Product Classifier - Usage Examples

## Quick Start

### 1. Prepare your data
```csv
name,fullGroupName
"iPhone 15 Pro Max","Accessories/Phone Cases/Apple iPhone 15 Pro Max"
"Samsung Galaxy S24","Accessories/Phone Cases/Samsung Galaxy S24"
```

### 2. Train model
```bash
python main.py
```

### 3. Test predictions
```bash
python testing_model.py
```

## Data Format

**Required columns:**
- `name` - product name (Russian/English)
- `fullGroupName` - category path with "/" separators

**Example:**
```csv
"iPhone 15 silicone case","Accessories/Phone Cases/Apple iPhone 15"
"Wireless headphones","Headphones/Apple/AirPods Pro"
```

## Commands

| Action | Command | Description |
|--------|---------|-------------|
| Train | `python main.py` | Train model with your data |
| Test | `python testing_model.py` | Test trained model |

| Install | `pip install -r requirements.txt` | Install dependencies |

## Expected Results

- **Accuracy**: 85-95%
- **Training time**: 15-30 seconds
- **Output files**: `class_model.h5`, `label_encoder.pkl`



## Troubleshooting

- Ensure CSV file is in project directory
- Check UTF-8 encoding for Russian text
- Minimum 3 products per category required
