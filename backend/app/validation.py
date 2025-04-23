from .model import get_feature_ranges

def validate_input(data):
    feature_ranges = get_feature_ranges()
    for feature in feature_ranges.keys():
        if feature not in data:
            return {
                "valid": False,
                "message": f"Missing feature: {feature}"
            }
    for feature, value in data.items():
        if feature not in feature_ranges:
            return {
                "valid": False,
                "message": f"Unknown feature: {feature}"
            }
        min_val, max_val = feature_ranges[feature]
        try:
            num_value = float(value)
            if num_value < min_val or num_value > max_val:
                return {
                    "valid": False,
                    "message": f"Feature {feature} with value {value} is outside the expected range ({min_val}, {max_val})"
                }
        except ValueError:
            return {
                "valid": False,
                "message": f"Feature {feature} has an invalid value: {value}. Numeric value expected."
            }
    return {"valid": True}
