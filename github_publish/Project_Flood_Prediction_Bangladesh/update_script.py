import re

path = r"c:\Users\Home\Documents\GitHub\Reasearch Model Implementation\Project_Flood_Prediction_Bangladesh\Hybrid_VAR_RF_Flood_Prediction.py"

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Delete Section 10 block entirely
# We know it starts at "10. PAPER-STYLE SUMMARY TABLE" and ends before "11. QUALITATIVE: PREDICTED"
import re
start_flag = "# ══════════════════════════════════════════════════════════════════════════════\n# 10. PAPER-STYLE SUMMARY TABLE"
end_flag = "# ══════════════════════════════════════════════════════════════════════════════\n# 11. QUALITATIVE"

idx1 = text.find(start_flag)
idx2 = text.find(end_flag)

if idx1 != -1 and idx2 != -1:
    text = text[:idx1] + text[idx2:]

# 2. Shift all section headers
text = text.replace("11. QUALITATIVE", "10. QUALITATIVE")
text = text.replace("divider(\"11. QUALITATIVE", "divider(\"10. QUALITATIVE")

text = text.replace("12. QUANTITATIVE", "11. QUANTITATIVE")
text = text.replace("divider(\"12. METRIC", "divider(\"11. METRIC")

text = text.replace("13. RAINY SEASON", "12. RAINY SEASON")
text = text.replace("divider(\"13. RAINY", "divider(\"12. RAINY")

text = text.replace("14. RESIDUAL", "13. RESIDUAL")
text = text.replace("divider(\"14. RESIDUAL", "divider(\"13. RESIDUAL")

text = text.replace("15. FLOOD RISK", "14. FLOOD RISK")
text = text.replace("divider(\"15. FLOOD", "divider(\"14. FLOOD")

text = text.replace("16. FORECAST", "15. FORECAST")
text = text.replace("divider(\"16. FORECAST", "divider(\"15. FORECAST")

text = text.replace("17. FINAL", "16. FINAL")
text = text.replace("divider(\"17. FINAL", "divider(\"16. FINAL")

# 3. Shift all plot names
text = text.replace("'07_", "'06_")
text = text.replace("'08_", "'07_")
text = text.replace("'09_", "'08_")
text = text.replace("'10_", "'09_")
text = text.replace("'11_", "'10_")
text = text.replace("'12_", "'11_")
text = text.replace("'13_", "'12_")
text = text.replace("'14_", "'13_")
text = text.replace("'15_", "'14_")

text = text.replace("All 15 output figures", "All 14 output figures")

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Done updating file.")
