from setuptools import setup, find_packages

setup(
    name = 'indoorContact',
    version = '0.1.0',
    description = "ABM pedestrian simulation counting indoor contact",
    url = 'https://github.com/MG-Choi/indoorCont.git',
    author = 'MoongiChoi',
    author_email = 'u1316663@utah.edu',
    packages = find_packages(),
    package_data = {'indoorContact': ['sampleData/obstacle_space.xlsx', 'sampleData/exp_space2.xlsx', 'sampleData/exp_space3.xlsx']},
    include_package_data = True,
    install_requires = ['matplotlib', 'pandas', 'numpy', 'tqdm', 'celluloid>=0.2.0', 'seaborn>=0.12.2']
)



'''
note: How to make library
- 모두 seqC -> py로 저장.

- cmd (administrator) -> cd repository
- python setup.py sdist bdist_wheel
- twine upload dist/*
- 업데이트시에는 setup.py -> 0.02로 하고 다시 위 과정 반복


library test는 cmd에서 한다.

- pip uninstall sequentPSS
- pip install sequentPSS


* 주의할 점:
random이나 os와 같이 깔려있는 library의 경우 위에 install_requires에 쓰지 않는다. py안에 바로 import로 쓰면 된다.

'''


#repository: C:\Users\MoongiChoi\Desktop\MG\양식, 코드 등\Python\Library\indoorCont



#참고:https://lsjsj92.tistory.com/592
#https://developer-theo.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-GitHub-Repository-%EC%83%9D%EC%84%B1%EB%B6%80%ED%84%B0-PyPI-Package-%EB%B0%B0%ED%8F%AC%EA%B9%8C%EC%A7%80

#위에서 버전 문제 발생: !pip install --upgrade requests
