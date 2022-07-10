from pydreamer.tools import load_npz

if __name__ == '__main__':
    npz_data_path = f'/home/yibo/Documents/pydreamer/mlruns/0/0158001.npz'
    data = load_npz(npz_data_path)
    for k, v in data.items():
        if k == 'image_pred':
            print(v[0][0].dtype)
