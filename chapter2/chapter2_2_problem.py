import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/bike_data.csv")

#1 계절(season)이 1일 때, 시간대(hour)별 총 대여량(count 합계)을 계산하고, 그 중 대여량이 가장 많은 시간대를 구하시오
#주석은 정답코드 주석 안한게 내가 푼 코드
#df['datetime'] = pd.to_datetime(df['datetime'])
#df['hour'] = df['datetime'].dt.hour
#result = df[df['season'] == 1].groupby('hour')['count'].sum()
#print(result.idxmax())

df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df_one = df[df['season'] == 1].copy()
df_one_sum = df_one.groupby('hour')['count'].sum().reset_index()
df_one_sum_max = df_one_sum['count'].idxmax()
print(df_one_sum.loc[df_one_sum_max,'hour'])
#2 각 계절(season)별 평균 대여량(count)를 구하시오
print(df.groupby('season')['count'].mean())
#3 1월(month)동안의 총 대여량(count)를 구하시오
df['month'] = df['datetime'].dt.month
total_jan = df[df['month'] == 1]['count'].sum()
print(total_jan)
#4 가장 대여량이 많은 날짜를 구하시오
df['date'] = df['datetime'].dt.date
# max_count = df.groupby('date')['count'].sum().reset_index()
# print(max_count['date'][max_count['count'].idxmax()])
daily_count = df.groupby('date')['count'].sum()
print(daily_count.idxmax())
#5 시간대(hour)별 평균 대여량(count)를 구하시오
print(df.groupby('hour')['count'].mean())
#6 월요일(weekday)동안의 총 대여량(count)를 구하시오
df['weekday'] = df['datetime'].dt.weekday
print(df[df['weekday'] == 0]['count'].sum())
#7 공유 자전거 데이터를 사용하여 넓은 형식(wide form)에서 긴 형식(long format)으로 변환하시오
# datatime과 season을 식별자(id_vars)로 유지하고, casual과 registered를 값 인자(value_vars)로 하여 데이터프레임을 생성하시오
long_format = pd.melt(df,id_vars=['datetime','season'],value_vars=['casual','registered'])
print(long_format)

#8 생성한 긴 형식 데이터 프레임을 활용하여 각 계절(season)별로 casual과 registered 사용자의 평균 대여 수(count)를 구하시오.
print(long_format.groupby(['season','variable'])['value'].mean())

df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/logdata.csv")

#9 로그 칼럼에서 연도 정보만 추출하시오
df['연도'] = df['로그'].str.extract(r'([0-9]+)')
print(df['연도'])
#10 로그 칼럼에서 '시:분:초'를 추출하시오
df['시분초'] = df['로그'].str.extract(r'([0-9]+:[0-9]+:[0-9]+)')
print(df['시분초'])
#11 로그 칼럼에서 한글 정보만 추출하시오
df['한글'] = df['로그'].str.replace(r'[^가-힣]','',regex=True)
print(df['한글'])
#12 로그 칼럼에서 특수 문자를 제거하시오
df['특수문자제거'] = df['로그'].str.replace(r'[^a-zA-Z0-9가-힣\s]','',regex=True)
print(df['특수문자제거'])
#13 로그 칼럼에서 유저,  Amount 값을 추출한 후, 각 유저별 Amount의 평균값을 계산하시오
print(df['로그'].iloc[1])
df['총량'] = df['로그'].str.extract(r'(Amount: +[\s0-9]+)')
df['총량'] = df['총량'].str.replace('Amount: ','')
df_drop = df.dropna()
df_drop['총량'] = df_drop['총량'].astype('int')
print(df_drop['총량'])
print(df_drop.groupby('한글')['총량'].mean())
