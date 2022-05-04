## Datasets
Download the dataset you want to train on and move to ./data

### Yelp 2013
https://figshare.com/articles/dataset/Yelp_2013/6292142
python train.py --dataset Yelp2013 --val_size 0.1 --test_size 0.1   --epochs 10 --create_tfrecord
### Yelp 2014
https://figshare.com/articles/dataset/Untitled_Item/6292253
python train.py --dataset Yelp2014 --val_size 0.1 --test_size 0.1   --epochs 10 --create_tfrecord
### Yelp 2015
https://figshare.com/articles/dataset/Yelp_2015/6292334
python train.py --dataset Yelp2015 --val_size 0.1 --test_size 0.1   --epochs 10 --create_tfrecord
### Amazon Review Full
https://doc-0o-88-docs.googleusercontent.com/docs/securesc/eiqg3i5fpjbgl3ab82cv1eump3meti6o/mve98b79ehsl0mv3j94vvr4s5lmr6v9t/1651620750000/07511006523564980941/07128768624675520494/0Bz8a_Dbh9QhbZVhsUnRWRDhETzA?resourcekey=0-Rp0ynafmZGZ5MflGmvwLGg&e=download&ax=ACxEAsa8ZR8QJT2gd3de9S6oq88rzVT-o_M32Rfhd6rTjW8wAAnU_y5cOuHsOcN4eteF45tVOKGplV0EVpD2axNOgnnYJ0y52dRDQJ8SGO7tkHD5B0jGQBYw2ZTn5JB1D6D5UGYusUFVlDCOKxFukjQYO1ZFpeH7EQGDsvJn-Bq7Y47dtt_TzVCUSFq1WznZxeFp1_AUSvYbSN3H99ut6zpU_NeOkfBqb0Gvl9fCYncqDG-ssFFJpPWAXEKAw2UjfKTpz6AyEw4zZNaw5cpg0LFdpg-oT5mbUDsPvWl0VVKf4JqH4rsE9f5WKsHOLB5sLtQKGnslF39UpVo2s_oV-SzfPhLtHUhZKquMIIPwTHwJ7XJ1lyEv325-9ZuEonTdFb_2ML7L07XCnr_JQvKrPCh5YSCthykCN2n5xeDjy9sq6AqKN3jlyF6m6_mfZn71-Pl_vfx7oLeeZBVSsnubuR4YcsDgXamjO-iNSm996dnWXDU_ms70J3IE6aACr21-bUnJe6gxgFlFdu7ryU5WhDSUduFWwpMouW_cA11rinWo6-ONfl6sfAL5abJao6FDtThBO8czX9q4lmIC9sbogcb-LKKfAS4hP03uYjoT-To2VISufgv5oL6eISMa8xHDwvwPz0O8yzSk33DcxC3mSIvc6WmIHR6YwNjDd4Njt-EhY2wFDjSi52Y9N24wxn9gvIY&authuser=0&nonce=voo3kggaghiga&user=07128768624675520494&hash=65tkv179vo82htotsv5msl94b2lab6c5

python train.py --create_tfrecord --dataset Custom --train_data_path data/amazon_review_full_csv/train.csv --test_data_path data/amazon_review_full_csv/test.csv --val_size 0.1 --epochs 20


# Result

Yelp 2013, batch size = 2048: 0.650
Yelp 2014, batch size = 2048: 0.689
Yelp 2015, batch size = 2048: 0.7195
