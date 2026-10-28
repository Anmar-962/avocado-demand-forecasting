# Dataset

`demand_sales_data.csv` is the original dataset supplied for the internship project. The author has confirmed that the provider permits its publication on GitHub. This statement does not grant a new dataset license or imply company endorsement.

The file is preserved unchanged. The analysis removes exact duplicates and reports excluded rows in `results/data_audit.json`.

Dates use month/day/year. The source mixes `case` and `kg` quantities. The model keeps case-unit rows only; kilogram rows remain in the full source CSV but are excluded from model aggregation. No case-to-kilogram conversion is provided or assumed. Company and customer identifiers are provided in the source data.

No external redistribution license was supplied. Contact the project author before redistributing the dataset beyond this repository.
