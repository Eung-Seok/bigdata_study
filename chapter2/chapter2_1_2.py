import pandas as pd
import numpy as np

# 펭귄 데이터 불러오고 기본 구조 확인
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/penguins.csv')
print(df.head())
print(df.tail())
print(df.describe())
df.info()

# 하나의 컬럼을 기준으로 정렬
sorted_df = df.sort_values(by='bill_length_mm', ascending=False)
print(sorted_df.head())

# 여러 컬럼을 기준으로 서로 다른 방향 정렬
sorted_df = df.sort_values(
    by=['bill_length_mm', 'bill_depth_mm'],
    ascending=[False, True]
)
print(sorted_df.head())

# idxmax, idxmin으로 최댓값/최솟값이 있는 행 찾기
max_idx = df['bill_length_mm'].idxmax()
print(f"maximum bill length index : {max_idx}")
print(df.loc[max_idx])

min_idx = df['bill_length_mm'].idxmin()
print(f"minimum bill length index : {min_idx}")
print(df.loc[min_idx])

# 정렬 후 첫 행 선택 방식과 idxmax 방식 비교
sorted_df = df.sort_values(by='bill_length_mm', ascending=False)
print("Sorted DataFrame by bill_length_mm (Descending) :")

sorted_max_value_row = sorted_df.iloc[0]
print("\nFirst row of the sorted DataFrame:")
print(sorted_max_value_row)

max_idx = df['bill_length_mm'].idxmax()
max_value_row = df.loc[max_idx]
print(f"\nMaximum bill length index using idxmax() : {max_idx}")
print(max_value_row)

comparison = max_value_row.equals(sorted_max_value_row)
print(f"\nDo the max value row match? {comparison}")

# species별로 묶어서 숫자형 컬럼 평균 계산
grouped_df = df.groupby('species').mean(numeric_only=True)
print(grouped_df)

# 전체 숫자형 컬럼의 평균과 합계 확인
print(df.mean(numeric_only=True))
print(df.sum(numeric_only=True))

# island별 flipper_length_mm 합계를 구하고 가장 큰 island 찾기
grouped_df = df.groupby('island')['flipper_length_mm'].sum()
print("Grouped by island and summed flipper_length_mm : ")
print(grouped_df)

max_flipper_islnad = grouped_df.idxmax()
print(f"\nIsland with maximum total flipper length : {max_flipper_islnad}")
print(grouped_df.loc[max_flipper_islnad])

# as_index=False는 그룹 기준 컬럼을 일반 컬럼으로 유지
grouped_sum_df = df.groupby('island', as_index=False)['flipper_length_mm'].sum()
print(grouped_sum_df)

sorted_grouped_sum_df = grouped_sum_df.sort_values(by='flipper_length_mm', ascending=False)
print("\nGrouped and sorted DataFrame by total flipper_length_mm:")
print(sorted_grouped_sum_df)

# merge 연습용 예제 데이터
df1 = pd.DataFrame({
    'key': ['A', 'B', 'C'],
    'value': [1, 2, 3]
})
df2 = pd.DataFrame({
    'key': ['A', 'B', 'D'],
    'value': [4, 5, 6]
})

# inner는 공통 key만, outer는 모든 key를 유지
merged_df = pd.merge(df1, df2, on='key', how='inner')
print(merged_df)

merged_df_outer = pd.merge(df1, df2, on='key', how='outer')
print(merged_df_outer)

# melt/pivot 연습용 wide 형태 예제 데이터
data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-03'],
    'Temperature': [10, 20, 25, 20],
    'Humidity': [60, 65, 70, 21]
}
df = pd.DataFrame(data)
print(df)

# melt는 wide 형태 데이터를 long 형태로 변환
df_melted = pd.melt(
    df,
    id_vars=['Date'],
    value_vars=['Temperature', 'Humidity'],
    var_name='Variable',
    value_name='Value'
)
print(df_melted)

# Date + Variable 조합이 중복되므로 여기서 pivot을 쓰면 오류 발생
# df_pivoted = df_melted.pivot(index='Date',
#                              columns='Variable',
#                              values='Value').reset_index()
# print(df_pivoted)

# 원래 각 행을 따로 유지해야 할 때 reset_index를 사용
df_melted2 = pd.melt(
    df.reset_index(),
    id_vars=['index'],
    value_vars=['Temperature', 'Humidity'],
    var_name='Variable',
    value_name='Value'
)
print(df_melted2)

df_pivoted = df_melted2.pivot(
    index='index',
    columns='Variable',
    values='Value'
)
print(df_pivoted)

# pivot_table은 중복값을 집계해서 처리 가능
df_pivot_table = df_melted.pivot_table(
    index='Date',
    columns='Variable',
    values='Value'
).reset_index()
print(df_pivot_table)

# rename, astype, apply, assign 연습용 학생 데이터 불러오기
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/dat.csv')
df.info()

# 컬럼명을 소문자로 변경
df = df.rename(columns={'Dalc': 'dalc', 'Walc': 'walc'})
print(df.columns)

# astype은 다시 저장하지 않으면 원본 df가 바뀌지 않음
df.astype({'famrel': 'object', 'dalc': 'float64'}).info()

# apply(axis=0/1) 연습용 간단한 예제 데이터
# df가 학생 데이터를 계속 가리키도록 주석 처리해 둠
# data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
# df = pd.DataFrame(data)
# print(df)
#
# print(df.apply(max, axis=0))
# print(df.apply(max, axis=1))

# 숫자형 가족 관계 점수를 문자 라벨로 변환하는 함수
def classify_famrel(famrel):
    if famrel <= 2:
        return 'Low'
    elif famrel <= 4:
        return 'Medium'
    else:
        return 'Hight'


# assign은 새 컬럼을 추가한 DataFrame을 반환
df1 = df.copy()
df1 = df1.assign(famrel_quality=df1['famrel'].apply(classify_famrel))
print(df1[['famrel', 'famrel_quality']].head())

# assign에서 기존 컬럼명을 쓰면 반환된 DataFrame 안에서 해당 컬럼이 덮어써짐
df2 = df.copy()
df2 = df2.assign(famrel=df2['famrel'].apply(classify_famrel))
print(df2[['famrel']].head())

# 직접 대입하면 복사한 DataFrame의 컬럼 자체가 변경됨
df3 = df.copy()
df3['famrel'] = df3['famrel'].apply(classify_famrel)
print(df3[['famrel']].head())

# 자료형 기준으로 컬럼 선택
print(df.select_dtypes('number').head(2))
print(df.select_dtypes('object').head(2))


# 숫자형 값을 컬럼별로 표준화
def standardize(x):
    return ((x - np.nanmean(x)) / np.std(x))


print(df.select_dtypes('number').apply(standardize).head(2))

# 컬럼명 패턴 기준으로 컬럼 선택
print(df.columns)
print(df.columns.str.startswith('f'))
print(df.loc[:, df.columns.str.startswith('f')].head())

print(df.columns.str.endswith('c'))
print(df.loc[:, df.columns.str.endswith('c')].head())

print(df.columns.str.contains('f'))
print(df.loc[:, df.columns.str.contains('f')].head())
print(df.loc[:, df.columns.str.contains('f')].apply(standardize).head())
