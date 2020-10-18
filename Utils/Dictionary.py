import numpy as np
aggregation = { #ZI Check Aggregation function is appropriate
    'Albumin': 'min',
    'Creatinine' : 'max',
    'C-Reactive-Protein' : 'max',
    'DiasBP' : 'min',
    'FiO2' : 'max',
    'Hb' : 'min',
    'Lymphocytes' : 'mean',
    'Neutrophils': 'mean',
    'NEWS2' : 'mean',
    'PLT': 'min',
    'PO2/FIO2' : 'min',
    'SysBP' : 'min',
    'Urea' : 'max',
    'WBC' : 'min/max'
}