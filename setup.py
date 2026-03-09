from setuptools import setup

setup(
  name = 'LTGN',         
  packages = ['LTGN'],  
  version = '0.0.3',      
  license= 'MIT',        
  description = 'Loss Trim Graph-based Network (LTGN), ' \
  'an unsupervised framework based on the interpretability of graph neural networks.',   
  author = 'Zedong Wang',                   
  author_email = 'wangzedong23@mails.ucas.ac.cn',     
  url = 'https://github.com/wzdrgi/LTGN',   
  keywords = ["scRNA-seq", 'GRN ', 'Deep Learning', 'Graph Neural Networks'],   
  install_requires=[           
          'scipy',  
          'numpy',
          'pandas',
          'scikit-learn',
          'pingouin',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',   
    'Programming Language :: Python :: 3.5', 
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
  ],
)
