import json
import statistics
from pathlib import Path
##single model
scores = [json.loads(x.joinpath('report.json').read_text())['metrics']['test']['score']
    for x in Path(
        '../exp/tabr_classGroup/rmse/car/612car_regression-evaluation'
    ).iterdir()
    if x.is_dir()]
print(f'{statistics.mean(scores)}\n\u00b1{statistics.stdev(scores)}','\n---------------')
print(f'{statistics.mean(scores):.4f}\u00b1{statistics.stdev(scores):.4f}','\n#########################')
#ensemble model
scores = [json.loads(x.joinpath('report.json').read_text())['metrics']['test']['score']
    for x in Path(
        '../exp/tabr_classGroup/rmse/car/612car_regression-ensemble-5'
    ).iterdir()
    if x.is_dir()]
print(f'{statistics.mean(scores)}\n\u00b1{statistics.stdev(scores)}','\n---------------')
print(f'{statistics.mean(scores):.4f}\u00b1{statistics.stdev(scores):.3f}','\n#########################')
