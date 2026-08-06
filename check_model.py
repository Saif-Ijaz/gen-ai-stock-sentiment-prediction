# check_model.py
import sys, os, joblib
sys.path.insert(0, '.')

bundle = joblib.load('models/model.pkl')
model  = bundle['model']
meta   = bundle.get('metadata', {})

print('=' * 50)
print('MODEL INFORMATION')
print('=' * 50)
print('Trained at: ', meta.get('trained_at', 'unknown'))
print('Accuracy:   ', meta.get('accuracy',   'unknown'))
print('F1 Score:   ', meta.get('f1',         'unknown'))
print('ROC-AUC:    ', meta.get('roc_auc',    'unknown'))
print('Precision:  ', meta.get('precision',  'unknown'))
print('Recall:     ', meta.get('recall',     'unknown'))
print('N Features: ', meta.get('n_features', 'unknown'))
print('Features:   ', meta.get('feature_names', 'unknown'))
print()
print('Model type:    ', type(model).__name__)
print('N estimators:  ', model.n_estimators)
print('Max depth:     ', model.max_depth)
print('Learning rate: ', model.learning_rate)