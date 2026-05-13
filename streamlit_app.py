"""
申请评分卡 — Streamlit 版本
线上部署：用户填写表单 → 模型预测 → 返回风险等级 + 信用评分
"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import sklearn
import lightgbm
import xgboost
import catboost

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title='申请评分卡 — 信用风险评估',
    page_icon='📊',
    layout='wide',
)

st.markdown("""
<style>
.main .block-container { padding-top: 2rem; }
.stButton > button {
    width: 100%; height: 3.5rem; font-size: 1.2rem; font-weight: 700;
    background: linear-gradient(135deg, #4361ee, #7209b7);
    color: white; border: none; border-radius: 10px; letter-spacing: 1px;
    transition: all 0.3s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(67,97,238,0.4); }
.risk-high { background: linear-gradient(135deg, #ff4d4f, #ff7875); padding: 2rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1rem; }
.risk-medium { background: linear-gradient(135deg, #faad14, #ffc53d); padding: 2rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1rem; }
.risk-low { background: linear-gradient(135deg, #52c41a, #73d13d); padding: 2rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1rem; }
.risk-label { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 加载流水线（缓存）
# ============================================================
@st.cache_resource
def load_pipeline():
    PIPELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scorecard_pipeline.pkl')
    with open(PIPELINE_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_pipeline()
fe_params = pipeline['fe_params']
final_lgb = pipeline['final_models']['lgb']
final_xgb = pipeline['final_models']['xgb']
final_cb = pipeline['final_models']['cb']
final_lr = pipeline['final_models']['lr']
final_scaler = pipeline['final_scaler']
weights = pipeline['blend_weights']
feature_cols = pipeline['feature_cols']

# ============================================================
# 默认值
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
# 特征工程
# ============================================================
def ensure_required_columns(df):
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
    df = df.copy()
    df = ensure_required_columns(df)

    le = fe_params['le_subGrade']
    df['subGrade'] = df['subGrade'].astype(str)
    known_classes = set(le.classes_)
    df['subGrade'] = df['subGrade'].apply(lambda x: x if x in known_classes else le.classes_[0])
    df['subGrade'] = le.transform(df['subGrade'])

    emp_map = fe_params['emp_len_map']
    df['employmentLength'] = df['employmentLength'].map(emp_map)

    df['earliesCreditLine_date'] = pd.to_datetime(df['earliesCreditLine'], format='%b-%Y', errors='coerce')
    ref_date = pd.Timestamp('2016-12-01')
    df['creditLineAge'] = ((ref_date - df['earliesCreditLine_date']).dt.days / 30).astype(float)
    df.drop('earliesCreditLine_date', axis=1, inplace=True)

    df['issueDate_dt'] = pd.to_datetime(df['issueDate'], errors='coerce')
    min_issue_date = pd.Timestamp('2007-01-01')
    df['issueDate_day'] = (df['issueDate_dt'] - min_issue_date).dt.days
    df['issueDate_month'] = df['issueDate_dt'].dt.month
    df['issueDate_year'] = df['issueDate_dt'].dt.year
    df['issueDate_quarter'] = df['issueDate_dt'].dt.quarter
    df.drop('issueDate_dt', axis=1, inplace=True)

    grade_map = fe_params['grade_map']
    df['grade'] = df['grade'].map(grade_map)

    for col in ['employmentLength', 'dti', 'revolUtil', 'pubRecBankruptcies',
                'annualIncome', 'openAcc', 'totalAcc', 'revolBal']:
        if col in df.columns:
            df[f'{col}_isnull'] = df[col].isnull().astype(int)
    df['missing_count'] = df[[c for c in df.columns if c.endswith('_isnull')]].sum(axis=1)

    n_cols = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
              'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']
    n_medians = fe_params.get('n_medians', {c: 0.0 for c in n_cols})
    for col in n_cols:
        df[col] = df[col].fillna(n_medians.get(col, 0.0))

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

    df['loanAmnt_log'] = np.log1p(df['loanAmnt'])
    df['annualIncome_log'] = np.log1p(df['annualIncome'])
    df['interestRate_log'] = np.log1p(df['interestRate'])
    df['dti_log'] = np.log1p(df['dti'].clip(lower=0))
    df['revolBal_log'] = np.log1p(df['revolBal'])
    df['installment_log'] = np.log1p(df['installment'])

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

    for col in ['employmentTitle', 'postCode', 'title']:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq)

    drop_cols = ['id', 'issueDate', 'earliesCreditLine', 'policyCode']
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)

    return df


def run_target_encoding(df):
    df = df.copy()
    te_cols = fe_params['te_cols']
    te_mappings = fe_params['te_mappings']
    global_mean = fe_params['global_mean']

    for col in te_cols:
        if col not in df.columns:
            continue
        mp = te_mappings[col]
        df[f'{col}_freq'] = df[col].map(mp['freq_train']).fillna(0)
        df[f'{col}_te'] = df[col].map(mp['smooth_all']).fillna(global_mean)
        df[f'{col}_nunique'] = df[col].map(mp['vc']).fillna(1)

    df.drop(te_cols, axis=1, inplace=True)
    return df


def fill_missing(df):
    fill_vals = fe_params['fill_values']
    for col in df.columns:
        if col in fill_vals:
            df[col] = df[col].fillna(fill_vals[col])
    return df


def align_columns(df):
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols]
    return df


def predict(df):
    df = run_feature_engineering(df)
    df = run_target_encoding(df)
    df = fill_missing(df)
    df = align_columns(df)

    p_lgb = final_lgb.predict_proba(df)[:, 1]
    p_xgb = final_xgb.predict_proba(df)[:, 1]
    p_cb = final_cb.predict_proba(df)[:, 1]
    df_scaled = final_scaler.transform(df)
    p_lr = final_lr.predict_proba(df_scaled)[:, 1]

    prob = (weights['lgb'] * p_lgb + weights['xgb'] * p_xgb +
            weights['cb'] * p_cb + weights['lr'] * p_lr)
    return prob


def prob_to_score(prob, base_score=600, base_odds=50, pdo=20):
    odds = (1 - prob) / prob
    score = base_score + pdo / np.log(2) * np.log(odds / base_odds)
    return np.clip(score, 300, 850)


def classify_risk(prob):
    if prob < 0.12:
        return '低风险', 'low', '#52c41a'
    elif prob < 0.25:
        return '中风险', 'medium', '#faad14'
    else:
        return '高风险', 'high', '#ff4d4f'


# ============================================================
# Streamlit UI
# ============================================================
st.title('📊 申请评分卡 — 信用风险评估')
st.markdown('填写贷款申请信息，系统将使用 **4模型融合**（LightGBM + XGBoost + CatBoost + LogisticRegression）预测违约概率并给出信用评分（300-850）')

# --- 侧边栏：模型信息 ---
with st.sidebar:
    st.header('📈 模型信息')
    st.metric('融合模型 AUC', '0.741+')
    st.metric('特征数量', len(feature_cols))
    st.metric('评分范围', '300 - 850')
    st.markdown('---')
    st.caption('基础分: 600 | PDO: 20 | Base Odds: 50')

col1, col2 = st.columns(2)

with col1:
    st.subheader('📋 贷款信息')
    loanAmnt = st.number_input('贷款金额 ($)', min_value=0.0, value=15000.0, step=100.0, key='loanAmnt')
    term = st.selectbox('贷款期限', [3, 5], index=1, format_func=lambda x: f'{x} 年', key='term')
    interestRate = st.number_input('利率 (%)', min_value=0.0, max_value=40.0, value=12.5, step=0.01, key='interestRate')
    installment = st.number_input('月供 ($)', min_value=0.0, value=500.0, step=1.0, key='installment')
    grade = st.selectbox('贷款等级', ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=2, key='grade')
    subGrade = st.selectbox('贷款子等级', [
        'A1','A2','A3','A4','A5','B1','B2','B3','B4','B5',
        'C1','C2','C3','C4','C5','D1','D2','D3','D4','D5',
        'E1','E2','E3','E4','E5','F1','F2','F3','F4','F5',
        'G1','G2','G3','G4','G5'
    ], index=10, key='subGrade')
    purpose = st.selectbox('贷款用途', ['债务重组', '其他'], index=1, key='purpose')
    purpose_map = {'债务重组': 0, '其他': 1}

with col2:
    st.subheader('👤 个人 & 收入信息')
    annualIncome = st.number_input('年收入 ($)', min_value=0.0, value=60000.0, step=100.0, key='annualIncome')
    employmentLength = st.selectbox('工作年限',
        ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
         '6 years', '7 years', '8 years', '9 years', '10+ years'],
        index=5, key='employmentLength')
    homeOwnership = st.selectbox('住房状况', ['租房', '自有住房', '按揭'], index=1, key='homeOwnership')
    homeOwnership_map = {'租房': 0, '自有住房': 1, '按揭': 2}
    verificationStatus = st.selectbox('收入验证状态', ['未验证', '已验证', '验证中'], index=1, key='verificationStatus')
    verificationStatus_map = {'未验证': 0, '已验证': 1, '验证中': 2}
    dti = st.number_input('债务收入比 DTI', min_value=0.0, max_value=100.0, value=18.5, step=0.01, key='dti')
    regionCode = st.number_input('地区代码', min_value=0, value=8, step=1, key='regionCode')

st.subheader('💳 信用历史')
col3, col4, col5 = st.columns(3)

with col3:
    ficoRangeLow = st.number_input('FICO 分下限', min_value=300, max_value=850, value=680, step=1, key='ficoRangeLow')
with col4:
    ficoRangeHigh = st.number_input('FICO 分上限', min_value=300, max_value=850, value=720, step=1, key='ficoRangeHigh')
with col5:
    earliesCreditLine = st.text_input('最早信用记录', value='Jan-2010', key='earliesCreditLine')

col6, col7, col8 = st.columns(3)
with col6:
    delinquency_2years = st.number_input('近2年逾期次数', min_value=0.0, max_value=50.0, value=0.0, step=1.0, key='delinquency_2years')
with col7:
    pubRec = st.number_input('公共记录数', min_value=0.0, value=0.0, step=1.0, key='pubRec')
with col8:
    pubRecBankruptcies = st.number_input('破产记录数', min_value=0.0, value=0.0, step=1.0, key='pubRecBankruptcies')

st.subheader('🏦 账户信息')
col9, col10, col11, col12 = st.columns(4)
with col9:
    openAcc = st.number_input('活跃账户数', min_value=0.0, value=8.0, step=1.0, key='openAcc')
with col10:
    totalAcc = st.number_input('总账户数', min_value=0.0, value=25.0, step=1.0, key='totalAcc')
with col11:
    revolBal = st.number_input('循环信用余额 ($)', min_value=0.0, value=8000.0, step=100.0, key='revolBal')
with col12:
    revolUtil = st.number_input('循环利用率 (%)', min_value=0.0, max_value=100.0, value=45.0, step=0.1, key='revolUtil')

# --- 提交按钮 ---
st.markdown('<br>', unsafe_allow_html=True)
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    submitted = st.button('🔍 开始评估信用风险', use_container_width=True)

# --- 处理预测 ---
if submitted:
    if loanAmnt <= 0 or annualIncome <= 0:
        st.error('请填写贷款金额和年收入（必须大于0）')
    else:
        with st.spinner('正在分析中，请稍候...'):
            row = {
                'loanAmnt': loanAmnt, 'term': term, 'interestRate': interestRate,
                'installment': installment, 'grade': grade, 'subGrade': subGrade,
                'purpose': purpose_map[purpose], 'annualIncome': annualIncome,
                'employmentLength': employmentLength, 'homeOwnership': homeOwnership_map[homeOwnership],
                'verificationStatus': verificationStatus_map[verificationStatus], 'dti': dti,
                'regionCode': regionCode, 'ficoRangeLow': ficoRangeLow, 'ficoRangeHigh': ficoRangeHigh,
                'delinquency_2years': delinquency_2years, 'pubRec': pubRec,
                'pubRecBankruptcies': pubRecBankruptcies, 'earliesCreditLine': earliesCreditLine,
                'openAcc': openAcc, 'totalAcc': totalAcc, 'revolBal': revolBal, 'revolUtil': revolUtil,
            }
            # 合并默认值
            full_row = {**DEFAULT_VALUES, **row}
            df = pd.DataFrame([full_row])

            prob = predict(df)[0]
            score = prob_to_score(prob)
            risk_label, risk_level, risk_color = classify_risk(prob)

        # --- 显示结果 ---
        st.markdown(f"""
        <div class="risk-{risk_level}">
            <div class="risk-label">{risk_label}</div>
            <p style="font-size:1.1rem; opacity:0.9;">违约概率 {prob*100:.2f}% | 信用评分 {score:.1f} 分</p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric('违约概率', f'{prob*100:.2f}%')
        with m2:
            st.metric('信用评分', f'{score:.1f}', help='范围 300-850')
        with m3:
            st.metric('模型融合 AUC', '0.741+')

        # 评分条
        pct = (score - 300) / (850 - 300)
        st.progress(float(np.clip(pct, 0.0, 1.0)))
        st.caption(f'评分条: 300 (极高风险) — 450 (高风险) — 550 (中风险) — 650 (低风险) — 850 (极低风险)')
