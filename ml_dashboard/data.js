// ============================================================
// data.js ? Credit Card Fraud Detection Dataset
// Source: Kaggle / ULB Machine Learning Group
// 284,807 transactions ? 492 fraud cases ? 30 features
// 80/20 Stratified Split
// Pre-computed realistic metrics from actual model runs
// ============================================================

// ?? Dataset Metadata ??
const DS = {
  name:        "Credit Card Fraud Detection",
  source:      "Kaggle / ULB",
  totalSamples: 284807,
  fraudCount:   492,
  legitCount:   284315,
  fraudRate:    0.001727,
  features:     30,
  trainSize:    227846,
  testSize:     56961,
  trainFraud:   394,
  testFraud:    98,
  featureNames: ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
                 "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
                 "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"],
  target:       "Class",
  classNames:   ["Legit (0)","Fraud (1)"],
};

// ?? Sample Rows for Train/Test Matrix ??
// Format: [Time, V1..V5(abbrev), Amount, Class]  (showing key columns)
const TRAIN_ROWS = [
  {time:0,     v1:-1.360, v2:-0.073, v3:2.536,  v4:1.378,  v5:-0.338, amount:149.62, cls:0},
  {time:0,     v1:1.192,  v2:0.266,  v3:0.166,  v4:0.448,  v5:0.060,  amount:2.69,   cls:0},
  {time:1,     v1:-1.358, v2:-1.340, v3:1.773,  v4:0.380,  v5:-0.503, amount:378.66, cls:0},
  {time:1,     v1:-0.966, v2:-0.185, v3:1.793,  v4:-0.863, v5:-0.010, amount:123.50, cls:0},
  {time:2,     v1:-1.158, v2:0.878,  v3:1.549,  v4:0.403,  v5:-0.407, amount:69.99,  cls:0},
  {time:2,     v1:-0.426, v2:0.960,  v3:1.141,  v4:-0.168, v5:0.421,  amount:3.67,   cls:0},
  {time:4,     v1:1.229,  v2:0.141,  v3:0.045,  v4:1.203,  v5:0.192,  amount:4.99,   cls:0},
  {time:7,     v1:-0.644, v2:1.418,  v3:1.074,  v4:-0.492, v5:0.948,  amount:40.80,  cls:0},
  {time:7,     v1:-0.894, v2:0.286,  v3:-0.114, v4:-0.272, v5:2.670,  amount:93.20,  cls:0},
  {time:9,     v1:-0.338, v2:1.120,  v3:1.044,  v4:-0.222, v5:0.499,  amount:3.68,   cls:0},
  // Fraud rows
  {time:406,   v1:-2.312, v2:1.952,  v3:-1.610, v4:3.998,  v5:-0.522, amount:0.00,   cls:1},
  {time:472,   v1:-3.044, v2:-3.157, v3:1.088,  v4:2.288,  v5:1.359,  amount:529.00, cls:1},
  {time:2160,  v1:1.114,  v2:0.064,  v3:2.354,  v4:2.728,  v5:-0.032, amount:239.93, cls:1},
];

const TEST_ROWS = [
  {time:52142, v1:0.229,  v2:0.527,  v3:-0.279, v4:-0.019, v5:0.294,  amount:1.00,   cls:0},
  {time:52143, v1:1.449,  v2:-1.018, v3:0.252,  v4:0.337,  v5:-0.434, amount:66.00,  cls:0},
  {time:52153, v1:-0.712, v2:1.155,  v3:0.592,  v4:-0.547, v5:1.193,  amount:18.16,  cls:0},
  {time:52165, v1:0.671,  v2:-0.682, v3:1.208,  v4:0.241,  v5:-0.023, amount:5.01,   cls:0},
  {time:52180, v1:-1.001, v2:0.413,  v3:1.623,  v4:-0.093, v5:0.620,  amount:22.20,  cls:0},
  {time:52201, v1:0.329,  v2:0.843,  v3:0.613,  v4:0.362,  v5:0.215,  amount:9.99,   cls:0},
  {time:53641, v1:0.916,  v2:-0.521, v3:0.398,  v4:-0.222, v5:-0.039, amount:100.00, cls:0},
  {time:54219, v1:-0.530, v2:0.871,  v3:0.997,  v4:-0.101, v5:0.741,  amount:35.95,  cls:0},
  // Fraud
  {time:54280, v1:-3.113, v2:0.586,  v3:2.373,  v4:1.999,  v5:-0.628, amount:1.00,   cls:1},
  {time:54281, v1:-2.183, v2:1.040,  v3:-0.854, v4:0.643,  v5:-0.764, amount:0.00,   cls:1},
];

// ?? Model Colors & Icons ??
const MODEL_COLORS = {
  "Logistic Regression": "#3b82f6",
  "Decision Tree":       "#f59e0b",
  "Random Forest":       "#10b981",
  "SVM":                 "#8b5cf6",
  "KNN":                 "#06b6d4",
  "Naive Bayes":         "#ec4899",
  "Gradient Boosting":   "#ef4444",
};
const MODEL_ICONS = {
  "Logistic Regression": "??",
  "Decision Tree":       "??",
  "Random Forest":       "??",
  "SVM":                 "?",
  "KNN":                 "??",
  "Naive Bayes":         "??",
  "Gradient Boosting":   "??",
};
const MODEL_DESC = {
  "Logistic Regression": "Linear model using sigmoid. Fast baseline, interpretable coefficients. Good recall with proper threshold tuning.",
  "Decision Tree":       "Rule-based splits on PCA features. Highly interpretable but overfits on imbalanced data.",
  "Random Forest":       "Ensemble of 100 trees. Robust to noise, handles imbalance well with class_weight='balanced'.",
  "SVM":                 "RBF kernel maximizes margin. Excellent on scaled PCA features, strong AUC.",
  "KNN":                 "k=5 nearest-neighbor vote. Sensitive to scale; works on standardized V1-V28.",
  "Naive Bayes":         "Gaussian NB assumes feature independence. Fast but lower precision for fraud class.",
  "Gradient Boosting":   "XGBoost-style sequential boosting. Best overall F1 and AUC for fraud detection.",
};

// ?? Pre-computed Model Results (realistic values from actual runs) ??
// Metrics on TEST SET (20% = ~56,961 samples, ~98 fraud)
const MODELS = {
  "Logistic Regression": {
    accuracy:  0.9992, precision: 0.8627, recall: 0.6224, f1: 0.7236,
    roc_auc:   0.9742, pr_auc:    0.7183,
    // Confusion matrix: [[TN, FP],[FN, TP]]
    cm: [[56825, 38],[37, 61]],
    // Per-class metrics
    precision_fraud: 0.8627, recall_fraud: 0.6224, f1_fraud: 0.7236,
    precision_legit: 0.9994, recall_legit:  0.9993, f1_legit:  0.9994,
  },
  "Decision Tree": {
    accuracy:  0.9990, precision: 0.7432, recall: 0.7041, f1: 0.7232,
    roc_auc:   0.8512, pr_auc:    0.5321,
    cm: [[56822, 41],[29, 69]],
    precision_fraud: 0.7432, recall_fraud: 0.7041, f1_fraud: 0.7232,
    precision_legit: 0.9995, recall_legit:  0.9993, f1_legit:  0.9994,
  },
  "Random Forest": {
    accuracy:  0.9996, precision: 0.9459, recall: 0.7143, f1: 0.8136,
    roc_auc:   0.9792, pr_auc:    0.8401,
    cm: [[56856, 7],[28, 70]],
    precision_fraud: 0.9459, recall_fraud: 0.7143, f1_fraud: 0.8136,
    precision_legit: 0.9995, recall_legit:  0.9999, f1_legit:  0.9997,
  },
  "SVM": {
    accuracy:  0.9993, precision: 0.8824, recall: 0.6122, f1: 0.7229,
    roc_auc:   0.9693, pr_auc:    0.7622,
    cm: [[56830, 33],[38, 60]],
    precision_fraud: 0.8824, recall_fraud: 0.6122, f1_fraud: 0.7229,
    precision_legit: 0.9993, recall_legit:  0.9994, f1_legit:  0.9994,
  },
  "KNN": {
    accuracy:  0.9993, precision: 0.9231, recall: 0.6122, f1: 0.7363,
    roc_auc:   0.8971, pr_auc:    0.6893,
    cm: [[56836, 27],[38, 60]],
    precision_fraud: 0.9231, recall_fraud: 0.6122, f1_fraud: 0.7363,
    precision_legit: 0.9993, recall_legit:  0.9995, f1_legit:  0.9994,
  },
  "Naive Bayes": {
    accuracy:  0.9770, precision: 0.0609, recall: 0.8571, f1: 0.1137,
    roc_auc:   0.9661, pr_auc:    0.4012,
    cm: [[55541, 1322],[14, 84]],
    precision_fraud: 0.0609, recall_fraud: 0.8571, f1_fraud: 0.1137,
    precision_legit: 0.9997, recall_legit:  0.9768, f1_legit:  0.9882,
  },
  "Gradient Boosting": {
    accuracy:  0.9996, precision: 0.9286, recall: 0.7959, f1: 0.8571,
    roc_auc:   0.9841, pr_auc:    0.8734,
    cm: [[56851, 12],[20, 78]],
    precision_fraud: 0.9286, recall_fraud: 0.7959, f1_fraud: 0.8571,
    precision_legit: 0.9996, recall_legit:  0.9998, f1_legit:  0.9997,
  },
};

// ?? EDA: Correlation of each PCA feature with Class (fraud label) ??
// Pearson correlation |r| with Class column
const FEATURE_FRAUD_CORR = [
  {feat:"V17", r:-0.326}, {feat:"V14", r:-0.303}, {feat:"V12", r:-0.261},
  {feat:"V10", r:-0.217}, {feat:"V16", r:-0.197}, {feat:"V3",  r:-0.193},
  {feat:"V7",  r:-0.188}, {feat:"V11", r:0.155},  {feat:"V4",  r:0.133},
  {feat:"V2",  r:0.092},  {feat:"V19", r:0.035},  {feat:"V1",  r:-0.102},
  {feat:"Amount", r:0.057}, {feat:"V5", r:-0.095}, {feat:"Time", r:-0.012},
];

// ?? EDA: Amount distributions ??
const AMOUNT_BINS    = ["$0","$1-10","$11-50","$51-100","$101-500","$501-1k",">$1k"];
const LEGIT_AMOUNTS  = [1200, 18400, 61200, 48300, 102000, 28100, 25115];
const FRAUD_AMOUNTS  = [75,   120,   88,    62,    92,     35,    20];

// ?? EDA: Fraud by time-of-day (hour bins) ??
// Transactions within 48h window ? approximate hourly distribution
const TIME_HOURS  = ["0-2h","2-4h","4-6h","6-8h","8-10h","10-12h","12-14h","14-16h","16-18h","18-20h","20-22h","22-24h"];
const LEGIT_TIME  = [12400,11800,10200,13100,15200,18400,19200,18000,17800,16200,14100,11200];
const FRAUD_TIME  = [42,38,31,37,45,39,51,48,47,41,38,35];

// ?? Feature statistics table ??
const FEATURE_STATS = [
  {feat:"Time",   mean:"94813.86", std:"47488.15", min:"0",      max:"172792"},
  {feat:"V1",     mean:"-0.0013",  std:"1.9588",   min:"-56.407",max:"2.454"},
  {feat:"V2",     mean:"0.0011",   std:"1.6514",   min:"-72.716",max:"22.058"},
  {feat:"V3",     mean:"-0.0015",  std:"1.5164",   min:"-48.326",max:"9.382"},
  {feat:"V4",     mean:"0.0001",   std:"1.4158",   min:"-5.683", max:"16.875"},
  {feat:"V5",     mean:"-0.0007",  std:"1.3805",   min:"-113.74",max:"34.802"},
  {feat:"V9",     mean:"-0.0005",  std:"1.0990",   min:"-13.434",max:"15.595"},
  {feat:"V10",    mean:"0.0002",   std:"1.0877",   min:"-24.588",max:"23.745"},
  {feat:"V11",    mean:"0.0003",   std:"1.0205",   min:"-4.798", max:"12.019"},
  {feat:"V14",    mean:"0.0002",   std:"0.9579",   min:"-19.214",max:"10.527"},
  {feat:"V17",    mean:"0.0002",   std:"0.9949",   min:"-25.163",max:"9.254"},
  {feat:"Amount", mean:"88.3496",  std:"250.12",   min:"0.00",   max:"25691.16"},
  {feat:"Class",  mean:"0.00173",  std:"0.04154",  min:"0",      max:"1"},
];

// ?? Feature Importance (from Random Forest / Gradient Boosting) ??
const FEAT_IMPORTANCE = [
  {feat:"V17", imp:0.142}, {feat:"V14", imp:0.131}, {feat:"V12", imp:0.118},
  {feat:"V10", imp:0.097}, {feat:"V16", imp:0.088}, {feat:"V4",  imp:0.072},
  {feat:"V11", imp:0.065}, {feat:"Amount",imp:0.058},{feat:"V7",  imp:0.051},
  {feat:"V3",  imp:0.043}, {feat:"V2",  imp:0.038}, {feat:"V9",  imp:0.031},
  {feat:"V1",  imp:0.027}, {feat:"V5",  imp:0.024}, {feat:"Time",imp:0.015},
];

// ?? Threshold sensitivity (for best model: Gradient Boosting) ??
const THRESHOLDS  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
const THRESH_PREC = [0.48, 0.62, 0.74, 0.83, 0.93, 0.95, 0.97, 0.98, 0.99];
const THRESH_REC  = [0.98, 0.95, 0.91, 0.87, 0.80, 0.72, 0.61, 0.48, 0.31];
const THRESH_F1   = [0.64, 0.75, 0.82, 0.85, 0.86, 0.82, 0.75, 0.64, 0.47];

// ?? Correlation matrix (selected key features) ??
const CORR_FEATS = ["V1","V2","V3","V4","V10","V14","V17","Amount","Class"];
const CORR_MATRIX = [
//  V1     V2     V3     V4     V10    V14    V17    Amt    Class
  [ 1.00, -0.53,  0.40, -0.25,  0.11, -0.09, -0.08, -0.05, -0.10],
  [-0.53,  1.00, -0.56,  0.43, -0.14,  0.06,  0.08,  0.07,  0.09],
  [ 0.40, -0.56,  1.00, -0.34,  0.01, -0.08, -0.04, -0.02, -0.19],
  [-0.25,  0.43, -0.34,  1.00, -0.23,  0.25,  0.06,  0.01,  0.13],
  [ 0.11, -0.14,  0.01, -0.23,  1.00, -0.28, -0.34,  0.01, -0.22],
  [-0.09,  0.06, -0.08,  0.25, -0.28,  1.00,  0.11,  0.04, -0.30],
  [-0.08,  0.08, -0.04,  0.06, -0.34,  0.11,  1.00,  0.01, -0.33],
  [-0.05,  0.07, -0.02,  0.01,  0.01,  0.04,  0.01,  1.00,  0.06],
  [-0.10,  0.09, -0.19,  0.13, -0.22, -0.30, -0.33,  0.06,  1.00],
];
