import uuid
import os
import math
import pandas as pd


def create_tmp_file(dir, suffix=None):
    filename = str(uuid.uuid4())
    if suffix is not None:
        filename += '.{}'.format(suffix)
    return os.path.join(dir, filename)


# center - x0,y0
def get_angle_in_circle(x, y, x0, y0):
    (dx, dy) = (x0-x, y-y0)
    # angle = atan(float(dy)/float(dx))
    angle = math.degrees(math.atan2(float(dy), float(dx)))
#     if angle < 0:
#         angle += 180
    return angle


def get_amit_anchors():
    with open('/export/home/ronlevy/projects/500FG_GTEx/amits_anchors_genes/selectedGenesForLupus.txt') as f:
        amits_filtered_data = f.readlines()
        amits_filtered_data = set([f.strip() for f in amits_filtered_data])

    with open('/export/home/ronlevy/projects/500FG_GTEx/amits_anchors_genes/selectedGenesForLupusLarge.txt') as f:
        amits_full_data = f.readlines()
        amits_full_data = set([f.strip() for f in amits_full_data])

    return amits_full_data, amits_filtered_data


def chunks(obj, chunk_size=10):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(obj), chunk_size):
        max_i = min(len(obj), i+chunk_size)
        # yield [obj[j] for j in range(i, max_i)]
        yield obj[i: max_i]


def fix_str(val):
    """
     runs on DF and makes str of tuple into a tuple of numbers
    """
    if pd.isna(val):
        return val
    else:
        return eval(val)


def fix_str_df(df):
    """
    run fix_str on df
    """
    return df.applymap(fix_str)


def rename_patient_name(df):
    """
    format GTEX naming to shorten, so individuals from several tissues could be easily joined
    """
    df.rename(columns=lambda x: 'GTEX-{}'.format(x.split('-')[1]), inplace=True)
    return df


def drop_na_by_pct(df, axis, pct_of_not_nulls=0.5, to_print=False):
    thresh = df.shape[axis]*pct_of_not_nulls
    ret_df = df.dropna(axis=1-axis, thresh=thresh)
    if to_print:
        print(df.shape, ret_df.shape)
    return ret_df


def make_pretty(val):
    """
    fit sub tissue value into subplots spacing..
    """
    sp = val.split(" ")
    if len(sp) == 1:
        return val
    elif len(sp) >= 3:
        return sp[0] + ' ' + sp[2][:10]
    else:
        return val
