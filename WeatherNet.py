import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('Data/amo_monthly.csv')
data = torch.tensor(data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']].values).flatten()
