import math
import pandas as pd


# center - x0,y0
def get_angle_in_circle(x, y, x0, y0):
    (dx, dy) = (x0-x, y-y0)
    # angle = atan(float(dy)/float(dx))
    angle = math.degrees(math.atan2(float(dy), float(dx)))
#     if angle < 0:
#         angle += 180
    return angle


def apply_angle(df, x_col, y_col, x0, y0,):
    return df.apply(lambda row: get_angle_in_circle(row[x_col], row[y_col], x0, y0), axis=1)


def get_amit_anchors():
    with open('/export/home/ronlevy/projects/500FG_GTEx/amits_anchors_genes/selectedGenesForLupus.txt') as f:
        amits_filtered_data = f.readlines()
        amits_filtered_data = set([f.strip() for f in amits_filtered_data])

    with open('/export/home/ronlevy/projects/500FG_GTEx/amits_anchors_genes/selectedGenesForLupusLarge.txt') as f:
        amits_full_data = f.readlines()
        amits_full_data = set([f.strip() for f in amits_full_data])

    return amits_full_data, amits_filtered_data


def rename_patient_name(df):
    """
    format GTEX naming to shorten, so individuals from several tissues could be easily joined
    """
    df.rename(columns=lambda x: 'GTEX-{}'.format(x.split('-')[1]), inplace=True)
    return df


def by_idx(val, idx):
    """
    return val[idx] in case val is not None
    :param val: indexed object - list/tuple
    :param idx: index to access
    :return: val[idx]
    """
    if pd.isna(val) or idx is None:
        return val
    return val[idx]


def assign_age_group(val):
    if val in ['50-59', '40-49']:
        return '40-60'
    elif val in ['60-69', '70-79']:
        return 'Above 60'
    elif val in ['30-39', '20-29']:
        return 'Below 40'
    else:
        raise
