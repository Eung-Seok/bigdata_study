import pandas as pd

# 날짜/시간 처리 연습용 예제 데이터
data = {
    'date' : ['2024-01-01 12:34:56', '2024-02-01 23:45:01', '2024-03-01 06:07:08', '2021-04-01 14:15:16'],
    'value' : [100,201,302,404]
}
df = pd.DataFrame(data)
df.info()

# 문자열 형태의 날짜를 datetime 자료형으로 변환
df['date'] = pd.to_datetime(df['date'])

print(df.head(2))
print(df.dtypes)

# format을 지정해서 다양한 날짜 문자열 변환
print(pd.to_datetime('02-2024-01', format='%m-%Y-%d'))
print(pd.to_datetime('2024년 01월 01일', format = '%Y년 %m월 %d일'))

# dt 접근자로 연/월/일 추출
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# 요일 이름과 요일 번호 추출
df['wday'] = df['date'].dt.day_name()
df['wday2'] = df['date'].dt.weekday

# 시/분/초 추출
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df['second'] = df['date'].dt.second

print(df['date'].dt.date)
print(df.head(2))

# 기준 날짜와의 일수 차이 계산
current_date = pd.to_datetime('2024-05-01')

df['days_diff'] = (current_date - df['date']).dt.days
print(df.head(2))

# 일정한 간격의 날짜 범위 생성
date_range = pd.date_range(start='2021-01-01', end='2021-01-10', freq='D')
print(date_range)

# year, month, day 컬럼을 조합해서 다시 datetime 생성
df['date2'] = pd.to_datetime(dict(year = df.year, month = df.month, day = df.day))
print(df[['date', 'date2']])

# 문자열 처리 연습용 예제 데이터
data = {
    '가전제품' : ['냉장고','세탁기','전자레인지','에어컨','청소기'],
    '브랜드' : ['LG','Samsung', 'Panasonic', 'Daikin', 'Dyson']
}
df = pd.DataFrame(data)

# 문자열 길이 계산
df['제품명_길이'] = df['가전제품'].str.len()
df['브랜드_길이'] = df['브랜드'].str.len()
print(df.head(2))

# 대소문자 변환
df['브랜드_소문자'] = df['브랜드'].str.lower()
df['브랜드_대문자'] = df['브랜드'].str.upper()
print(df[['브랜드','브랜드_소문자','브랜드_대문자']])

# 특정 문자열 포함 여부 확인
df['브랜드에_a포함'] = df['브랜드'].str.contains('a')
print(df['브랜드에_a포함'])

# 문자열 일부 교체
df['브랜드_교체'] = df['브랜드'].str.replace('L','HHHHG')
print(df[['브랜드','브랜드_교체']])

# 특정 문자를 기준으로 문자열 분리
df[['브랜드_첫부분', '브랜드_두번째', '브랜드_세번째']] = df['브랜드'].str.split('a', expand=True)
print(df[['브랜드','브랜드_첫부분','브랜드_두번째','브랜드_세번째']])

# 두 문자열 컬럼 연결
df['제품_브랜드'] = df['가전제품'].str.cat(df['브랜드'], sep=', ')
print(df[['가전제품','제품_브랜드']])

# 앞뒤 공백 제거
df['가전제품'] = df['가전제품'].str.replace('전자레인지', '  전자레인지  ')
df['가전제품_공백제거'] = df['가전제품'].str.strip()
print(df[['가전제품','가전제품_공백제거']])

# 문자열 길이를 맞추기 위해 왼쪽/오른쪽에 문자 채우기
df["브랜드_pad"] = df['브랜드'].str.pad(width=10, side='left',fillchar='0')
df['브랜드_pad_right'] = df['브랜드'].str.pad(width=10,side='right',fillchar='*')
print(df[['브랜드','브랜드_pad','브랜드_pad_right']])

# 정규식 처리 연습용 주소 데이터
data = {
    '주소' : ['서울특별시 강남구 테헤란로 123', '부산광역시 해운대구 센텀중앙로 45', '대구광역시 수성구 동대구로 77-9@@##', '인천광역시 남동구 예술로 501&&, 아트센터', '광주광역시 북구 용봉로 123']
}
df = pd.DataFrame(data)
print(df)

# extract는 괄호로 캡처한 부분만 추출
df['도시'] = df['주소'].str.extract(r'([가-힣]+광역시|[가-힣]+특별시)', expand=False)
print(df)

# extractall은 정규식에 맞는 모든 값을 행 형태로 추출
special_chars = df['주소'].str.extractall(r'([^a-zA-Z0-9가-힣\s])')
print(special_chars)

# 정규식으로 특수문자를 찾아 빈 문자열로 교체
df['주소_특수문자제거'] = df['주소'].str.replace(r'([^a-zA-Z0-9가-힣\s])', '', regex=True)
print(df[['주소','주소_특수문자제거']])
