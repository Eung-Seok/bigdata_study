import numpy as np
import pandas as pd

# 기본 DataFrame 생성
df = pd.DataFrame({
    'col1': ['one', 'two'],
    'col2': [6, 7]
})

# 컬럼명, 자료형, 행/열 크기 확인
print(df)
print(df.columns)
print(df['col1'].dtype)
print(df['col2'].dtype)
print(df.shape)

# CSV 파일에서 시험 점수 데이터 불러오기
url = "https://raw.githubusercontent.com/YoungjinBD/data/main/examscore.csv"
mydata = pd.read_csv(url)

print(mydata.head())
print(mydata.shape)

# 인덱스와 이름을 직접 지정해서 Series 생성
data = [10, 20, 30]
df_s = pd.Series(data, index=["one", "two", "three"], name="count")
print(df_s.dtype)
print(df_s.shape)
print(df_s.name)
print(df_s)

# 컬럼명으로 열 선택
print(mydata['gender'])
print(mydata[['gender', 'midterm']])

# 조건에 맞는 행 필터링
print(mydata[mydata['midterm'] <= 15])

# iloc으로 숫자 위치 기준 행/열 선택
print(mydata.iloc[:, 0].head(2))
print(mydata.iloc[:, 0].shape)
print(type(mydata.iloc[:, 0]))

# iloc[:, 0]은 Series, iloc[:, [0]]은 DataFrame으로 반환
print(mydata.iloc[:, [0]].shape)
print(type(mydata.iloc[:, [0]]))

print(mydata.iloc[:, [0, 1]].head(2))

# squeeze는 1열짜리 DataFrame을 Series로 줄일 수 있음
print(mydata.iloc[:, [0]])
print(type(mydata.iloc[:, [0]]))
print(type(mydata.iloc[:, [0]].squeeze()))

# 같은 조건 필터링을 기본 인덱싱과 loc 방식으로 비교
print(mydata[mydata['midterm'] <= 15])
print(mydata.loc[mydata['midterm'] <= 15])
print(mydata.loc[mydata['midterm'] <= 15, :])
print(mydata.loc[mydata['midterm'] <= 15, ['student_id', 'final']])

# 조건이 여러 개일 때는 괄호와 함께 &, |, ~ 사용
print(mydata.loc[(mydata['gender'] == 'F') & (mydata['midterm'] >= 30), ['student_id', 'final']])

# isin은 값이 목록 안에 포함되는지 확인
print(mydata[mydata['midterm'].isin([28, 38, 52])].head())
print(mydata.loc[mydata['midterm'].isin([28, 38, 52]), ['student_id', 'final']].head())
print(mydata.loc[~mydata['midterm'].isin([28, 38, 52])].head())


# 결측치 연습용 작은 예제 데이터
mydata = pd.DataFrame({
    'student_id': [1, 2, 3, 4, 5],
    'gender': ['F', 'M', 'F', 'M', 'M'],
    'midterm': [38, 42, 53, 48, 46],
    'final': [46, 67, 56, 54, 39]
})

# 일부 값을 직접 결측치로 변경
mydata.iloc[0, 1] = np.nan
mydata.iloc[4, 0] = np.nan
print(mydata.head())

# 컬럼별 결측치 개수 확인
print("gender 열의 결측치 개수: ", mydata['gender'].isna().sum())
print("student_id 열의 결측치 개수: ", mydata['student_id'].isna().sum())

# 결측치가 하나라도 있는 행 제거
complete_row = mydata.dropna()
print("완전한 행의 수: ", len(complete_row))
print(complete_row)

# 컬럼 추가 및 삭제
mydata['total'] = mydata['midterm'] + mydata['final']
print(mydata.iloc[0:3, [3, 4]])

del mydata['gender']
print(mydata.head())

# concat 연습용 예제 DataFrame
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
})

df2 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5']
})

df3 = pd.DataFrame({
    'C': ['C0', 'C1', 'C2'],
    'D': ['D0', 'D1', 'D2']
})

# 행 방향으로 이어붙이기
result = pd.concat([df1, df2])
print(result)

# 열 방향으로 이어붙이기
result = pd.concat([df1, df3], axis=1)
print(result)

result = pd.concat([df1, df2])
print(result)

# 이어붙인 뒤 인덱스 새로 부여
result = pd.concat([df1, df2], ignore_index=True)
print(result)

df4 = pd.DataFrame({
    'A': ['A2', 'A3', 'A4'],
    'B': ['B2', 'B3', 'B4'],
    'C': ['C2', 'C3', 'C4']
})

print(df1)
print(df4)

# inner는 공통 컬럼만 유지, outer는 모든 컬럼 유지
result = pd.concat([df1, df4], join='inner')
print(result)

result = pd.concat([df1, df4], join='outer')
print(result)

# keys를 사용하면 계층형 인덱스 생성
result = pd.concat([df1, df2], keys=['key1', 'key2'])
print(result)

# key2 그룹을 선택한 뒤 위치 기준으로 일부 행 선택
df1_rows = result.loc['key2'].iloc[1:3]
print(df1_rows)
