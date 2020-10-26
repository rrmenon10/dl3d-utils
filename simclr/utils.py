import yaml
from easydict import EasyDict as edict

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'

import seaborn as sns
import scipy.misc
sns.set_style('white')


def load_config(path):
    with open(path) as fin:
        config = edict(yaml.safe_load(fin))

    return config

def plot_loss(loss, 
			  title='',
			  path='default.png'):

	plt.figure()
	plt.plot(range(len(loss)), loss)
	plt.title(title)
	plt.savefig(path, dpi=300)
	plt.tight_layout()
	plt.close()