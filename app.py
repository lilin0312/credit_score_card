"""
申请评分卡 — Flask 后端 API
加载训练好的流水线，接收 JSON 输入，返回违约概率 + 信用评分 + 风险等级
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ============================================================
# 加载流水线
# ============================================================
PIPELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scorecard_pipeline.pkl')
print(f'Loading pipeline from: {PIPELINE_PATH}')
with open(PIPELINE_PATH, 'rb') as f:
    pipeline = pickle.load(f)

fe_params = pipeline['fe_params']
final_lgb = pipeline['final_models']['lgb']
final_xgb = pipeline['final_models']['xgb']
final_cb = pipeline['final_models']['cb']
final_lr = pipeline['final_models']['lr']
final_scaler = pipeline['final_scaler']
weights = pipeline['blend_weights']
feature_cols = pipeline['feature_cols']
print(f'Pipeline loaded. {len(feature_cols)} features.')

# ============================================================
# 默认值（用于用户未填写的字段）
# ============================================================
DEFAULT_VALUES = {
    'term': 3,
    'installment': 500.0,
    'employmentTitle': 0.0,
    'employmentLength': '5 years',
    'homeOwnership': 0,
    'verificationStatus': 1,
    'purpose': 0,
    'postCode': 0.0,
    'regionCode': 0,
    'delinquency_2years': 0.0,
    'openAcc': 5.0,
    'pubRec': 0.0,
    'pubRecBankruptcies': 0.0,
    'revolBal': 10000.0,
    'revolUtil': 50.0,
    'totalAcc': 20.0,
    'initialListStatus': 0,
    'applicationType': 0,
    'policyCode': 1.0,
    'n0': 0.0, 'n1': 0.0, 'n2': 0.0, 'n3': 0.0, 'n4': 0.0,
    'n5': 0.0, 'n6': 0.0, 'n7': 0.0, 'n8': 0.0, 'n9': 0.0,
    'n10': 0.0, 'n11': 0.0, 'n12': 0.0, 'n13': 0.0, 'n14': 0.0,
    'earliesCreditLine': 'Jan-2010',
    'issueDate': 'Jan-2016',
    'title': 0.0,
}

# ============================================================
# 特征工程（复现 notebook 的 add_features + target encoding）
# ============================================================
def ensure_required_columns(df):
    """确保所有必需列存在，缺失的填默认值"""
    required = ['loanAmnt', 'term', 'interestRate', 'installment', 'grade',
                'subGrade', 'employmentTitle', 'employmentLength', 'homeOwnership',
                'annualIncome', 'verificationStatus', 'issueDate', 'purpose',
                'postCode', 'regionCode', 'dti', 'delinquency_2years',
                'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec',
                'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc',
                'initialListStatus', 'applicationType', 'earliesCreditLine',
                'title', 'policyCode',
                'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9',
                'n10', 'n11', 'n12', 'n13', 'n14']
    for col in required:
        if col not in df.columns:
            df[col] = DEFAULT_VALUES.get(col, 0)
    return df


def run_feature_engineering(df):
    """对单行/多行原始数据执行完整特征工程"""
    df = df.copy()
    df = ensure_required_columns(df)

    # --- subGrade 编码 ---
    le = fe_params['le_subGrade']
    df['subGrade'] = df['subGrade'].astype(str)
    # 处理未见过的类别
    known_classes = set(le.classes_)
    df['subGrade'] = df['subGrade'].apply(
        lambda x: x if x in known_classes else le.classes_[0]
    )
    df['subGrade'] = le.transform(df['subGrade'])

    # --- employmentLength 映射 ---
    emp_map = fe_params['emp_len_map']
    df['employmentLength'] = df['employmentLength'].map(emp_map)

    # --- 日期特征 ---
    df['earliesCreditLine_date'] = pd.to_datetime(df['earliesCreditLine'], format='%b-%Y', errors='coerce')
    ref_date = pd.Timestamp('2016-12-01')
    df['creditLineAge'] = ((ref_date - df['earliesCreditLine_date']).dt.days / 30).astype(float)
    df.drop('earliesCreditLine_date', axis=1, inplace=True)

    df['issueDate_dt'] = pd.to_datetime(df['issueDate'], errors='coerce')
    # issueDate_day 用训练集的最小日期作为基准（保存在 fe_params 中）
    min_issue_date = pd.Timestamp('2007-01-01')  # fallback
    df['issueDate_day'] = (df['issueDate_dt'] - min_issue_date).dt.days
    df['issueDate_month'] = df['issueDate_dt'].dt.month
    df['issueDate_year'] = df['issueDate_dt'].dt.year
    df['issueDate_quarter'] = df['issueDate_dt'].dt.quarter
    df.drop('issueDate_dt', axis=1, inplace=True)

    # --- grade 映射 ---
    grade_map = fe_params['grade_map']
    df['grade'] = df['grade'].map(grade_map)

    # --- 缺失值指示器 ---
    for col in ['employmentLength', 'dti', 'revolUtil', 'pubRecBankruptcies',
                'annualIncome', 'openAcc', 'totalAcc', 'revolBal']:
        if col in df.columns:
            df[f'{col}_isnull'] = df[col].isnull().astype(int)

    df['missing_count'] = df[[c for c in df.columns if c.endswith('_isnull')]].sum(axis=1)

    # --- n列填充 ---
    n_cols = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
              'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']
    n_medians = fe_params.get('n_medians', {c: 0.0 for c in n_cols})
    for col in n_cols:
        df[col] = df[col].fillna(n_medians.get(col, 0.0))

    # --- 核心比率 ---
    df['fico_mean'] = (df['ficoRangeLow'] + df['ficoRangeHigh']) / 2
    df['fico_range'] = df['ficoRangeHigh'] - df['ficoRangeLow']
    df['totalPayment'] = df['installment'] * df['term']
    df['loanAmnt_income_ratio'] = df['loanAmnt'] / (df['annualIncome'] + 1)
    df['installment_income_ratio'] = df['installment'] / (df['annualIncome'] + 1)
    df['interest_loanAmnt'] = df['interestRate'] * df['loanAmnt']
    df['revolBal_income_ratio'] = df['revolBal'] / (df['annualIncome'] + 1)
    df['openAcc_totalAcc_ratio'] = df['openAcc'] / (df['totalAcc'] + 1)
    df['delinquency_fico'] = df['delinquency_2years'] * df['fico_mean']
    df['dti_income'] = df['dti'] * df['annualIncome']
    df['revolUtil_fico'] = df['revolUtil'] * df['fico_mean']
    df['loanAmnt_totalPayment_ratio'] = df['loanAmnt'] / (df['totalPayment'] + 1)
    df['grade_interestRate'] = df['grade'] * df['interestRate']
    df['fico_interestRate'] = df['fico_mean'] * df['interestRate']
    df['fico_dti'] = df['fico_mean'] * df['dti']
    df['revolBal_loanAmnt_ratio'] = df['revolBal'] / (df['loanAmnt'] + 1)
    df['grade_dti'] = df['grade'] * df['dti']
    df['grade_loanAmnt'] = df['grade'] * df['loanAmnt']
    df['term_interestRate'] = df['term'] * df['interestRate']
    df['subGrade_interestRate'] = df['subGrade'] * df['interestRate']
    df['creditLineAge_fico'] = df['creditLineAge'] * df['fico_mean']
    df['verificationStatus_interestRate'] = df['verificationStatus'] * df['interestRate']
    df['installment_term_income_ratio'] = df['totalPayment'] / (df['annualIncome'] + 1)
    df['revolUtil_dti'] = df['revolUtil'] * df['dti']
    df['fico_grade'] = df['fico_mean'] * df['grade']
    df['delinquency_pubRec'] = df['delinquency_2years'] + df['pubRec']
    df['creditLineAge_dti'] = df['creditLineAge'] * df['dti']
    df['creditLineAge_income'] = df['creditLineAge'] * df['annualIncome']
    df['openAcc_loanAmnt_ratio'] = df['openAcc'] / (df['loanAmnt'] + 1)
    df['totalAcc_income_ratio'] = df['totalAcc'] / (df['totalAcc'] + 1)
    df['installment_loanAmnt_ratio'] = df['installment'] / (df['loanAmnt'] + 1)
    df['interestRate_income'] = df['interestRate'] * df['annualIncome']
    df['employmentLength_income'] = df['employmentLength'] * df['annualIncome']
    df['regionCode_interestRate'] = df['regionCode'] * df['interestRate']
    df['homeOwnership_income'] = df['homeOwnership'] * df['annualIncome']
    df['delinquency_openAcc'] = df['delinquency_2years'] * df['openAcc']
    df['pubRec_totalAcc_ratio'] = df['pubRec'] / (df['totalAcc'] + 1)
    df['fico_homeOwnership'] = df['fico_mean'] * df['homeOwnership']
    df['term_loanAmnt_income'] = df['term'] * df['loanAmnt_income_ratio']

    # --- 风险评分 ---
    df['risk_score'] = df['grade'] * df['interestRate'] * df['loanAmnt_income_ratio']
    df['repay_capacity'] = df['annualIncome'] / (df['installment'] + 1)
    df['income_after_debt'] = df['annualIncome'] * (1 - df['dti'] / 100)
    df['income_after_debt_to_loan'] = df['income_after_debt'] / (df['loanAmnt'] + 1)
    df['overpay_ratio'] = df['totalPayment'] / (df['loanAmnt'] + 1)
    df['income_to_fico'] = df['annualIncome'] / (df['fico_mean'] + 1)
    df['interestRate_to_fico'] = df['interestRate'] / (df['fico_mean'] + 1)
    df['dti_to_fico'] = df['dti'] / (df['fico_mean'] + 1)
    df['closeAcc'] = df['totalAcc'] - df['openAcc']
    df['closeAcc_ratio'] = df['closeAcc'] / (df['totalAcc'] + 1)
    df['delinquency_ratio'] = df['delinquency_2years'] / (df['creditLineAge'] / 12 + 1)
    df['credit_history_length'] = df['creditLineAge']

    # --- 对数特征 ---
    df['loanAmnt_log'] = np.log1p(df['loanAmnt'])
    df['annualIncome_log'] = np.log1p(df['annualIncome'])
    df['interestRate_log'] = np.log1p(df['interestRate'])
    df['dti_log'] = np.log1p(df['dti'].clip(lower=0))
    df['revolBal_log'] = np.log1p(df['revolBal'])
    df['installment_log'] = np.log1p(df['installment'])

    # --- n列统计 ---
    df['n_sum'] = df[n_cols].sum(axis=1)
    df['n_std'] = df[n_cols].std(axis=1)
    df['n_max'] = df[n_cols].max(axis=1)
    df['n_min'] = df[n_cols].min(axis=1)
    df['n_median'] = df[n_cols].median(axis=1)
    df['n_skew'] = df[n_cols].skew(axis=1)
    df['n_range'] = df['n_max'] - df['n_min']
    df['n0_n2_ratio'] = df['n0'] / (df['n2'] + 1)
    df['n4_n10_sum'] = df['n4'] + df['n10']
    df['n0_n1_diff'] = df['n0'] - df['n1']
    df['n2_n3_diff'] = df['n2'] - df['n3']
    df['n_sum_grade'] = df['n_sum'] * df['grade']
    df['n8_interestRate'] = df['n8'] * df['interestRate']
    df['n2_loanAmnt'] = df['n2'] * df['loanAmnt']
    df['n_std_dti'] = df['n_std'] * df['dti']

    # --- 频率编码 ---
    for col in ['employmentTitle', 'postCode', 'title']:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq)

    # --- 删除无用列 ---
    drop_cols = ['id', 'issueDate', 'earliesCreditLine', 'policyCode']
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)

    return df


def run_target_encoding(df):
    """对特征工程后的数据执行 target encoding"""
    df = df.copy()
    te_cols = fe_params['te_cols']
    te_mappings = fe_params['te_mappings']
    global_mean = fe_params['global_mean']

    for col in te_cols:
        if col not in df.columns:
            continue
        mp = te_mappings[col]

        # 频率编码
        df[f'{col}_freq'] = df[col].map(mp['freq_train']).fillna(0)

        # target encoding
        df[f'{col}_te'] = df[col].map(mp['smooth_all']).fillna(global_mean)

        # nunique 编码
        df[f'{col}_nunique'] = df[col].map(mp['vc']).fillna(1)

    df.drop(te_cols, axis=1, inplace=True)
    return df


def fill_missing(df):
    """用训练集的中位数填充缺失值"""
    fill_vals = fe_params['fill_values']
    for col in df.columns:
        if col in fill_vals:
            df[col] = df[col].fillna(fill_vals[col])
    return df


def align_columns(df):
    """确保列顺序与训练时一致"""
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols]
    return df


# ============================================================
# 预测函数
# ============================================================
def predict(df):
    """完整预测流水线: 特征工程 → 编码 → 预测 → 评分"""
    # 特征工程
    df = run_feature_engineering(df)
    # Target encoding
    df = run_target_encoding(df)
    # 填充缺失
    df = fill_missing(df)
    # 对齐列
    df = align_columns(df)

    # 模型预测
    p_lgb = final_lgb.predict_proba(df)[:, 1]
    p_xgb = final_xgb.predict_proba(df)[:, 1]
    p_cb = final_cb.predict_proba(df)[:, 1]
    df_scaled = final_scaler.transform(df)
    p_lr = final_lr.predict_proba(df_scaled)[:, 1]

    # 加权融合
    prob = (weights['lgb'] * p_lgb + weights['xgb'] * p_xgb +
            weights['cb'] * p_cb + weights['lr'] * p_lr)

    return prob


def prob_to_score(prob, base_score=600, base_odds=50, pdo=20):
    """违约概率 → 信用评分 (300-850)"""
    odds = (1 - prob) / prob
    score = base_score + pdo / np.log(2) * np.log(odds / base_odds)
    return np.clip(score, 300, 850)


def classify_risk(prob):
    """根据违约概率判定风险等级"""
    if prob < 0.12:
        return '低风险', 'low', '#52c41a'
    elif prob < 0.25:
        return '中风险', 'medium', '#faad14'
    else:
        return '高风险', 'high', '#ff4d4f'


# ============================================================
# API 路由
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '请求体为空，请提供JSON数据'}), 400

        # 合并默认值
        row = {}
        for k, v in DEFAULT_VALUES.items():
            row[k] = data.get(k, v)
        # 用户提供的值覆盖默认值
        row.update(data)

        # 转为 DataFrame
        df = pd.DataFrame([row])

        # 预测
        prob = predict(df)[0]
        score = prob_to_score(prob)
        risk_label, risk_level, risk_color = classify_risk(prob)

        return jsonify({
            'success': True,
            'probability': round(float(prob), 6),
            'probability_pct': f'{prob*100:.2f}%',
            'score': round(float(score), 1),
            'risk_level': risk_level,
            'risk_label': risk_label,
            'risk_color': risk_color,
            'base_score': 600,
            'score_range': [300, 850],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'features': len(feature_cols)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
