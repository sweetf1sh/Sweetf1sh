import streamlit as st
import numpy as np
import scipy.optimize as sco
import scipy.interpolate as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from scipy.stats import norm
import time
import requests
from dateutil.relativedelta import relativedelta

# ファンドコードとAPIで取得した対応表から対応するファンド名を表示
def Fundname(fundcode, code_list):
  record = code_list.query('fund_cd == @fundcode')
  return record.filter(items = ['fund_name']).iloc[-1, -1]

# ファンドの基準価額を取得
def get_nav_fund(fundcode, use_ratio=False):
    """
    fundcode: market code of each target fund (ex. "NNNNNN") defined by each asset management firm.
    start, end: datetime object
    """
    # Get fund data from https://www.am.mufg.jp/
    # time.sleep(1) # アクセス規制を避けるため
    url = "https://www.am.mufg.jp/fund_file/setteirai/"+ fundcode +".csv"
    df = pd.read_csv(url, encoding="shift_jis", skiprows=1)
    df['基準日'] = pd.to_datetime(df['基準日'])
    df = df.set_index('基準日')   
    if use_ratio:
        df = df.pct_change()
    return df

# ファンドの基準価額一覧表を作成
def get_paneldata_fund(fundcodes, start, end, use_ratio=False):
    """
    fundcode: market code of each target fund (ex. "NNNNNN") defined by each asset management firm.
    start, end: datetime object
    """
    # Use "基準価額（分配金再投資）(円)	" value only 
    dfs = []
    for fc in fundcodes:
        df = get_nav_fund(fc, use_ratio)[['基準価額（分配金再投資）(円)']]
        df = df.rename(columns={'基準価額（分配金再投資）(円)': str(fc)})
        dfs.append(df)
    df_concat = pd.concat(dfs, axis=1)
    df_concat.index = pd.to_datetime(df_concat.index)
    df_concat = df_concat[start:end] # 期間を範囲指定
    return df_concat

# リスク寄与度の計算
def calculate_risk_contribution(w, Sigma):
  w = np.matrix(w)
  sigma_p =  np.dot(w.T, np.dot(Sigma, w))
  RC_i = np.multiply(Sigma*w.T, w.T)/sigma_p
  return RC_i

# 目的関数の設定
## 最小分散
def minimize_vol(w):
    return np.dot(w.T, np.dot(Sigma, w)) #σ_p^2 = w'Vw

## シャープレシオ最大化
## 目的関数は最小化されるのでマイナスをつける
def maximize_SR(w):
    return - (np.dot(Mu, w) - rf) / np.sqrt(np.dot(w.T, np.dot(Sigma, w))) #σ_p^2 = w'Vw

## リスクパリティ（誤差二乗和の最小化）
def risk_budget_objective(w):
  sigma_p = np.dot(w.T, np.dot(Sigma, w))
  risk_target = np.asmatrix(np.multiply(sigma_p, rc_t))
  RC_i = calculate_risk_contribution(w, sigma_p)
  J = sum(np.square(RC_i - risk_target.T))[0,0] 
  return J

# セッションデータ自体が存在しない場合、新たに作成して値を初期化
st.session_state['flag'] = False

# ファンドコードとファンド名の対応表を取得
url = "https://developer.am.mufg.jp/code_list"
response = requests.get(url)
code_list = response.json()
code_list = pd.DataFrame(code_list['datasets'])

st.title("ポートフォリオ最適化ツール") # タイトル

st.sidebar.header("ファンドリターン取得")
st.sidebar.write("三菱UFJアセットマネジメントの投資信託に割り振られている6桁の数字を入力するとファンドを変更できます。")
FundCd1 = st.sidebar.text_input("ファンド1", '252634') # 初期値はｅＭＡＸＩＳ Ｓｌｉｍ 国内株式（ＴＯＰＩＸ） 
FundCd2 = st.sidebar.text_input("ファンド2", '252653') # 初期値はｅＭＡＸＩＳ Ｓｌｉｍ 先進国株式インデックス 
FundCd3 = st.sidebar.text_input("ファンド3", '252878') # 初期値はｅＭＡＸＩＳ Ｓｌｉｍ 新興国株式インデックス 
FundCd4 = st.sidebar.text_input("ファンド4", '252648') # 初期値はｅＭＡＸＩＳ Ｓｌｉｍ 国内債券インデックス 
FundCd5 = st.sidebar.text_input("ファンド5", '252667') # 初期値はｅＭＡＸＩＳ Ｓｌｉｍ 先進国債券インデックス 
FundCd6 = st.sidebar.text_input("ファンド6", '260448') # 初期値はｅＭＡＸＩＳ 新興国債券インデックス 
FundCd7 = st.sidebar.text_input("ファンド7", '253669') # 初期値はｅＭＡＸＩＳ Ｓｌｉｍ 国内リートインデックス 
FundCd8 = st.sidebar.text_input("ファンド8", '253674') # 初期値はｅＭＡＸＩＳ Ｓｌｉｍ 先進国リートインデックス 
FundCd9 = st.sidebar.text_input("ファンド9", '260894') # 初期値はｅＭＡＸＩＳ 新興国リートインデックス
FundCd10 = st.sidebar.text_input("ファンド10", '251065') #初期値は三菱ＵＦＪ 純金ファンド
st.sidebar.write("")
start = st.sidebar.date_input('計算開始日', value=datetime.date.today() + relativedelta(years=-5), min_value=datetime.date(1900, 1, 1))
end = st.sidebar.date_input('計算終了日', value=datetime.date.today(), min_value=datetime.date(1900, 1, 1))

NAVdata = st.sidebar.button("データ取得") # データ取得ボタン

#ファンドコードとファンド名をリストにまとめておく
FundCds = [FundCd1, FundCd2, FundCd3, FundCd4, FundCd5, FundCd6, FundCd7, FundCd8, FundCd9, FundCd10]
FundNames = []
for fc in FundCds:
        FundNamei = Fundname(fc, code_list)
        FundNames.append(FundNamei)

if NAVdata:
    return_df = get_paneldata_fund(FundCds, start, end, use_ratio=True) * 100 # 百分率に換算
    st.session_state.return_df = return_df #取得データを保持
    st.sidebar.write("データ取得完了")

st.header("各ファンドの過去リターン・リスク・相関")
try:
    # 取得データを読み込み
    return_df = st.session_state.return_df
    st.dataframe(pd.DataFrame({
        'ファンド名' : FundNames
       ,'過去リターン（年率）' : return_df.mean().values * 250
       , '過去リスク（年率）' : return_df.std().values * np.sqrt(250)
    }))
    st.session_state.correl = st.dataframe(return_df.corr())
    st.session_state.covar = st.dataframe(return_df.cov()*250)
except:
    st.write("サイドバーのデータ取得ボタンを押してファンドのリターンデータを取得してください。")
    
st.header("ポートフォリオ最適化")
rf = st.number_input('預金金利（％）', min_value=-10.00, max_value=100.00, value=0.01, step=0.01)
st.write("期待リターン・ウェイト上下限")

method = st.radio("最適化方法", ("シャープレシオ最大化", "最小分散", "リスクパリティ"))

try:
    # 取得データを読み込み
     return_df = st.session_state.return_df
     # 3カラム表示
     col= st.columns([1, 1, 1])
     expret1 = col[0].number_input(FundNames[0] + ": 期待リターン（％）", min_value=-10.00, max_value=100.00, value=return_df.mean().values[0] * 250, step=1.00)
     lower_bound1 = col[1].number_input(FundNames[0] + "下限",min_value=0.00, max_value=100.00, value=0.00, step=1.00)
     upper_bound1 = col[2].number_input(FundNames[0] + "上限",min_value=0.00, max_value=100.00, value=100.00, step=1.00)
     expret2 = col[0].number_input(FundNames[1] + ": 期待リターン（％）", min_value=-10.00, max_value=100.00, value=return_df.mean().values[1] * 250, step=1.00)
     lower_bound2 = col[1].number_input(FundNames[1] + "下限",min_value=0.00, max_value=100.00, value=0.00, step=1.00)
     upper_bound2 = col[2].number_input(FundNames[1] + "上限",min_value=0.00, max_value=100.00, value=100.00, step=1.00)
     expret3 = col[0].number_input(FundNames[2] + ": 期待リターン（％）", min_value=-10.00, max_value=100.00, value=return_df.mean().values[2] * 250, step=0.01)
     lower_bound3 = col[1].number_input(FundNames[2] + "下限",min_value=0.00, max_value=100.00, value=0.00, step=1.00)
     upper_bound3 = col[2].number_input(FundNames[2] + "上限",min_value=0.00, max_value=100.00, value=100.00, step=1.00)
     expret4 = col[0].number_input(FundNames[3] + ": 期待リターン（％）", min_value=-10.00, max_value=100.00, value=return_df.mean().values[3] * 250, step=0.01)
     lower_bound4 = col[1].number_input(FundNames[3] + "下限",min_value=0.00, max_value=100.00, value=0.00, step=1.00)
     upper_bound4 = col[2].number_input(FundNames[3] + "上限",min_value=0.00, max_value=100.00, value=100.00, step=1.00)
     expret5 = col[0].number_input(FundNames[4] + ": 期待リターン（％）", min_value=-10.00, max_value=100.00, value=return_df.mean().values[4] * 250, step=0.01)
     lower_bound5 = col[1].number_input(FundNames[4] + "下限",min_value=0.00, max_value=100.00, value=0.00, step=1.00)
     upper_bound5 = col[2].number_input(FundNames[4] + "上限",min_value=0.00, max_value=100.00, value=100.00, step=1.00)
     expret6 = col[0].number_input(FundNames[5] + ": 期待リターン（％）", min_value=-10.00, max_value=100.00, value=return_df.mean().values[5] * 250, step=0.01)
     lower_bound6 = col[1].number_input(FundNames[5] + "下限",min_value=0.00, max_value=100.00, value=0.00, step=1.00)
     upper_bound6 = col[2].number_input(FundNames[5] + "上限",min_value=0.00, max_value=100.00, value=100.00, step=1.00)
     expret7 = col[0].number_input(FundNames[6] + ": 期待リターン（％）", min_value=-10.00, max_value=100.00, value=return_df.mean().values[6] * 250, step=0.01)
     lower_bound7 = col[1].number_input(FundNames[6] + "下限",min_value=0.00, max_value=100.00, value=0.00, step=1.00)
     upper_bound7 = col[2].number_input(FundNames[6] + "上限",min_value=0.00, max_value=100.00, value=100.00, step=1.00)
     expret8 = col[0].number_input(FundNames[7] + ": 期待リターン（％）", min_value=-10.00, max_value=100.00, value=return_df.mean().values[7] * 250, step=0.01)
     lower_bound8 = col[1].number_input(FundNames[7] + "下限",min_value=0.00, max_value=100.00, value=0.00, step=1.00)
     upper_bound8 = col[2].number_input(FundNames[7] + "上限",min_value=0.00, max_value=100.00, value=100.00, step=1.00)
     expret9 = col[0].number_input(FundNames[8] + ": 期待リターン（％）", min_value=-10.00, max_value=100.00, value=return_df.mean().values[8] * 250, step=0.01)
     lower_bound9 = col[1].number_input(FundNames[8] + "下限",min_value=0.00, max_value=100.00, value=0.00, step=1.00)
     upper_bound9 = col[2].number_input(FundNames[8] + "上限",min_value=0.00, max_value=100.00, value=100.00, step=1.00)
     expret10 = col[0].number_input(FundNames[9] + ": 期待リターン（％）", min_value=-10.00, max_value=100.00, value=return_df.mean().values[9] * 250, step=0.01)
     lower_bound10 = col[1].number_input(FundNames[9] + "下限",min_value=0.00, max_value=100.00, value=0.00, step=1.00)
     upper_bound10 = col[2].number_input(FundNames[9] + "上限",min_value=0.00, max_value=100.00, value=100.00, step=1.00)
     
     # 初期値の設定
     x0 = [1. / len(return_df.columns)] * len(return_df.columns) # 等分とする
     # 目的関数に必要な定数の準備
     Mu = [expret1, expret2, expret3, expret4, expret5, expret6, expret7, expret8, expret9, expret10]
     Sigma = return_df.cov().values * 250 # 各資産の分散共分散行列
     rc_t = [1. / len(return_df.columns)] * len(return_df.columns) # ターゲットのリスク寄与度
     # 制約条件
     ## ウェイト合計は1
     cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
     bnds = [
         (lower_bound1, upper_bound1) 
         ,(lower_bound2, upper_bound2) 
         ,(lower_bound3, upper_bound3) 
         ,(lower_bound4, upper_bound4) 
         ,(lower_bound5, upper_bound5) 
         ,(lower_bound6, upper_bound6) 
         ,(lower_bound7, upper_bound7) 
         ,(lower_bound8, upper_bound8) 
         ,(lower_bound9, upper_bound9) 
         ,(lower_bound10, upper_bound10) 
             ]
except:
    st.write("サイドバーのデータ取得ボタンを押してファンドのリターンデータを取得してください。")

optimize = st.button("実行") # 最適化実行ボタン

if optimize:
     # 最適化実行
     if method == "シャープレシオ最大化":
         opts = sco.minimize(fun=maximize_SR, x0=x0, method='SLSQP', bounds=bnds, constraints=cons)
     elif method == "最小分散":
         opts = sco.minimize(fun=minimize_vol, x0=x0, method='SLSQP', bounds=bnds, constraints=cons)
     elif method == "リスクパリティ":
         opts = sco.minimize(fun=risk_budget_objective, x0=x0, method='SLSQP', bounds=bnds, constraints=cons)
     # 計算結果を保持
     st.session_state.opts = opts
     # 最適解を円グラフとして表示
     st.write("期待リターン（％）: " + str(np.dot(Mu, opts["x"])))
     st.write("推定リスク（％）: " + str(np.sqrt(minimize_vol(opts["x"]))))
     st.write("シャープレシオ: " + str(-maximize_SR(opts["x"])))
     fig, ax = plt.subplots()
     #plt.rcParams["font.family"] = "Yu Gothic"
     plt.pie(opts["x"], labels=FundNames, autopct="%1.1f %%")
     st.pyplot(fig) 

st.header("預貯金考慮後期待リターン・推定リスク")
     
rf_amount = st.number_input('預貯金合計金額', min_value=0, max_value=5000000000000000, value=1000, step=1000)
port_amount = st.number_input('ポートフォリオ合計金額', min_value=0, max_value=5000000000000000, value=0, step=1000)

try:
    # 取得データを読み込み
    opts = st.session_state.opts
    st.write("ポートフォリオウェイト（％）")
    weight1 = st.number_input(FundNames[0], min_value=0.00, max_value=100.00, value=opts["x"][0]*100, step=0.01)
    weight2 = st.number_input(FundNames[1], min_value=0.00, max_value=100.00, value=opts["x"][1]*100, step=0.01)
    weight3 = st.number_input(FundNames[2], min_value=0.00, max_value=100.00, value=opts["x"][2]*100, step=0.01)
    weight4 = st.number_input(FundNames[3], min_value=0.00, max_value=100.00, value=opts["x"][3]*100, step=0.01)
    weight5 = st.number_input(FundNames[4], min_value=0.00, max_value=100.00, value=opts["x"][4]*100, step=0.01)
    weight6 = st.number_input(FundNames[5], min_value=0.00, max_value=100.00, value=opts["x"][5]*100, step=0.01)
    weight7 = st.number_input(FundNames[6], min_value=0.00, max_value=100.00, value=opts["x"][6]*100, step=0.01)
    weight8 = st.number_input(FundNames[7], min_value=0.00, max_value=100.00, value=opts["x"][7]*100, step=0.01)
    weight9 = st.number_input(FundNames[8], min_value=0.00, max_value=100.00, value=opts["x"][8]*100, step=0.01)
    weight10 = st.number_input(FundNames[9], min_value=0.00, max_value=100.00, value=opts["x"][9]*100, step=0.01)
    weights = [weight1 / 100, weight2 / 100, weight3 / 100, weight4 / 100, weight5 / 100, weight6 / 100, weight7 / 100, weight8 / 100, weight9 / 100, weight10 / 100]
    ex_port_ret = rf_amount / (port_amount + rf_amount) * rf + port_amount / (port_amount + rf_amount) * np.dot(Mu, weights)
    ex_port_risk = port_amount / (port_amount + rf_amount) * np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
    st.write("期待リターン（％）: " + str(ex_port_ret))
    st.write("推定リスク（％）: " + str(ex_port_risk))
    st.write("リターンが正規分布に従うと仮定すると" +  str(norm.cdf(-2, loc=0, scale=1)*100) + "％の確率で" + str(int(-(ex_port_ret - 2*ex_port_risk) * (port_amount + rf_amount)/100)) + "円の損失が発生する可能性が想定されます。")
except:
    st.write("上の実行ボタンを押して最適化計算を実行してください。")

st.header("参考にしたwebサイト")
st.write("ファンド名取得部分: https://www.am.mufg.jp/tool/webapi/")
st.write("データ取得部分: https://qiita.com/faux/items/d709cc9b43f18df70382")
st.write("最適化部分: https://qiita.com/geoha6/items/4a3da038140fc5bdfce6")
st.write("webアプリ（Streamlit）部分: https://www.alpha.co.jp/blog/202304_02")

