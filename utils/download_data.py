import gdown
import os

root_path = os.path.dirname(os.path.abspath(__file__)).split('utils')[:-1][0]

def download_data():
    path = os.path.join(root_path, 'data/total_data/bird_data_total.pkl')
    if os.path.isfile(path) is False:
        print('bird data file does not exist. Download mavik data from google drive ... ')

        bird_url = 'https://drive.google.com/file/d/1pvbS9xXntgImjmUQT8Dx49cjJ7ZmaQGf/view?usp=sharing'
        output_path = path
        gdown.download(bird_url, output_path, quiet=False,fuzzy=True)

    #path = 'sj_work/data/total_cut_data/mavik_data_total.pkl'
    path = os.path.join(root_path, 'data/total_data/mavik_data_total.pkl')
    if os.path.isfile(path) is False:
        print('mavik data file does not exist. . Download mavik data from google drive ... ')
        mavik_url = 'https://drive.google.com/file/d/1eA2NbjCxZupnCqJ99wKsXeafrlU-LqkQ/view?usp=sharing'
        output_path = path
        gdown.download(mavik_url, output_path, quiet=False,fuzzy=True)

