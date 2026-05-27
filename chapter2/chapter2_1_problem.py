import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/grade.csv")

#1 df 데이터 프레임의 정보를 출력하고, 각 칼럼의 데이터 타입을 확인하시오.
df.info()
#2 midterm 점수가 85점 이상인 학생들의 데이터를 필터링하여 출력하시오
print(df[df['midterm']>=85])
#3 final 점수를 기준으로 데이터 프레임을 내림차순으로 정렬하고, 정렬된 데이터 프레임의 첫 5행을 출력하시오.
print(df.sort_values(by='final',ascending=False).head())
#4 gender 칼럼을 기준으로 데이터 프레임을 그룹화하고, 각 그룹별 midterm과 final의 평균을 계산하여 출력하시오.
#주석된 풀이로 써서 결과는 나왔는데 문법적으로는 틀렸음
# df_group = df.groupby('gender').mean()
# print(df_group[['midterm','final']])
df_group = df.groupby('gender')[['midterm','final']].mean()
print(df_group)
#5 student_id 칼럼을 문자열 타입으로 변환하고, 변환된 데이터 프레임의 정보를 출력하시오.
df['student_id'] = df['student_id'].astype('str')
df.info()

#6 assignment 점수의 최댓값과 최솟값을 가지는 행을 각각 출력하시오
ass_max_idx = df['assignment'].idxmax()
ass_min_idx = df['assignment'].idxmin()
print(df.iloc[ass_max_idx])
print(df.iloc[ass_min_idx])
#7 midterm, final, assignment 점수의 평균을 계산하여 average 칼럼을 추가하고, 첫 5행을 출력하시오.
df['average'] = df[['assignment','midterm','final']].mean(axis=1)
print(df.head())
#8 데이터 프레임에 결측치가 있는지 확인하고, 결측치를 포함한 행을 제거한 후 데이터 프레임의 정보를 출력하시오.
nan_sum = df.isna().sum()
print(nan_sum)
df = df.dropna()
df.info()
#9 아래의 추가 데이터를 생성하고, 기존 데이터 프레임과 student_id를 기준으로 병합하여 출력하시오
additional_data = {
    'student_id' : ['1','3','5','7','9'],
    'club' : ['Art','Science','Math','Music','Drama']
}
df_additional = pd.DataFrame(additional_data)

print(pd.merge(df,df_additional,on='student_id',how='inner'))
#10 gender를 인덱스로, student_id를 열로 사용하여 average 점수에 대한 피벗 테이블을 생성하고 출력하시오.
print(pd.pivot_table(df, index='gender',columns='student_id',values='average'))
#11 midterm, final, assignment의 평균을 구하고, average 열을 생성하시오, 또한 성별, 성적 유형(assignment, average, final, midterm)별 평균 점수를 계산하시오.
#아예 손도 못대서 그냥 정답보고 따라 썼음
df_melted = pd.melt(df, id_vars=['student_id','name','gender'],value_vars=['assignment','average','final','midterm'],var_name='variable',value_name='score')
grouped_mean = df_melted.groupby(['gender','variable'])['score'].mean().reset_index()
print(grouped_mean)

#12 최대 평균 성적을 가진 학생의 이름과 평균 성적을 출력하시오.
max_average_idx = df['average'].idxmax()
print(df.iloc[max_average_idx][['name','average']])
