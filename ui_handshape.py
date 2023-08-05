from __future__ import annotations
import numpy.typing as npt

import streamlit as st

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from PIL import Image
from visualize import get_hand_points
from semisupervision import cluster_frames, get_distances_kmeans
from glob import glob
import joblib


############### Constants ####################
HANDSHAPE_IMAGES_DIR = 'handshape-images'
HANDSHAPE_CSV = 'handshapes.csv'
DATA_DIR = '../ASL-FSR/data'
##############################################


st.set_page_config(layout='wide')


##########################################
# Revtrieved via https://aslfont.github.io/Symbol-Font-For-ASL/asl/handshapes.html
handshapes = pd.read_csv(HANDSHAPE_CSV)
NO_SELECTION_STR = '--SELECT--'
selection_list = [NO_SELECTION_STR] + handshapes['gloss'].to_list()
##########################################

st.title('Labeler')

user_name = st.text_input(
    label='Name of User (Labeler)',
    value=os.environ.get('COMP_NAME'),
)

@st.cache_data
def load_selection_images():
    images = {
        fname.split('/')[-1].split('.png')[0]: Image.open(fname)
        for fname in glob(f'{HANDSHAPE_IMAGES_DIR}/*.png')
    }
    return images

@st.cache_data
def load_data(sample_size: int | None = None) -> pd.DataFrame:
    df_data = pd.read_csv(f'{DATA_DIR}/train.csv')
    pq_path = df_data['path'].unique()[0]
    df_pq = pd.read_parquet(f'{DATA_DIR}/{pq_path}')
    # Remove everything except the right hand
    data = df_pq[
        [
            c for c in df_pq.columns 
            if (
                ('x_right' in c)
                or ('y_right' in c)
                or ('frame' in c)
            )
        ]
    ]
    # Allow for a sample
    if sample_size:
        data = data.sample(sample_size)
    # Remove null-values and get a NumPy array
    data = data.fillna(0)
    data = data.join(
        df_data[['phrase','sequence_id']].set_index('sequence_id'),
        how='left',
    )
    print(data.shape)
    return data



@st.cache_data
def read_dict(file_path):
    path = os.path.expanduser(file_path)
    with open(path, 'r') as f:
        d = json.load(f)
    return d


# Load data
# data = load_data(sample_size=1_000) # TODO: Allow sample size by choosing
data_full = load_data(sample_size=None)
data = (
    data_full
    .drop(
        [
            'phrase',
            'frame',
        ],
        axis=1,
    )
)
st.write(data.head())


def get_representative_images(
    frames: npt.ArrayLike,
    kmeans = None,
    frame_subset_mask = None,
    **cluster_kwargs,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
# ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    '''Get representative images from frames after clustering.

    Defaults to KMeans to be calculated but `n_clusters` must be defined first
    '''
    if frame_subset_mask is None:
        frame_subset_mask = np.ones(frames.shape[0], dtype='bool')

    frames_subset = frames[frame_subset_mask]

    kmeans_dist, _ = get_distances_kmeans(
        frames=frames_subset,
        kmeans=kmeans,
        **cluster_kwargs,
    )
    representative_frame_idx = np.argmin(kmeans_dist, axis=0)
    #
    all_frame_idx = []
    for i,s in enumerate(frame_subset_mask):
        if s: 
            all_frame_idx.extend([
                    i * frames.shape[1] + j
                    for j in range(frames.shape[1])
            ])
    all_frame_idx = np.array(all_frame_idx)
    subset_frame_idx = all_frame_idx[representative_frame_idx]
    
    representative_frames = frames_subset[representative_frame_idx]

    return representative_frames, subset_frame_idx

HAND_PARTS_DEFINITION = [
    np.array([0,1,2,3,4,]), # thumb
    np.array([5,6,7,8,]), # index
    np.array([9,10,11,12,]), # f2
    np.array([13,14,15,16,]), # f3
    np.array([17,18,19,20,]), # f4
    np.array([0,5,9,13,17,0,]), # palm
]

def get_hand_points(handx, handy, parts_definitions=HAND_PARTS_DEFINITION):
    x = [handx[part_def] for part_def in parts_definitions]
    y = [handy[part_def] for part_def in parts_definitions]
    return x, y


def plot_handshape(
    frame: npt.ArrayLike,
    ax: plt.Axes,
    **plot_kwargs,
) -> None:
    buffer = 0.1
    xmin = np.nanmin(frame[[col for col in frame.columns if ('x_' in col and 'pose' not in col)]]) - buffer
    xmax = np.nanmax(frame[[col for col in frame.columns if ('x_' in col and 'pose' not in col)]]) + buffer
    ymax = -1*(np.nanmin(frame[[col for col in frame.columns if ('y_' in col and 'pose' not in col)]])) + buffer
    ymin = -1*(np.nanmax(frame[[col for col in frame.columns if ('y_' in col and 'pose' not in col)]])) - buffer

    x_min_max = (xmin,xmax)
    y_min_max = (ymin,ymax)

    ########
    right_x = frame[[c for c in frame.columns if 'x_right' in c]].values
    right_y = frame[[c for c in frame.columns if 'y_right' in c]].values * -1

    # left_x = frame[[c for c in frame.columns if 'x_left' in c]].values
    # left_y = frame[[c for c in frame.columns if 'y_left' in c]].values * -1

    # face_x = frame[[c for c in frame.columns if 'x_face' in c]].values
    # face_y = frame[[c for c in frame.columns if 'y_face' in c]].values * -1

    # TODO: Include pose points in viz
    # pose_x = frame[[c for c in frame.columns if 'x_pose' in c]].values
    # pose_y = frame[[c for c in frame.columns if 'y_pose' in c]].values * -1

    # Note that ax will be defined outside of function
    # ax.clear()
    # ax.plot(face_x, face_y, ',', color='green',)
    # ax.plot(pose_x[0, :], pose_y[0, :], '.', color='black')

    temp_x, temp_y = get_hand_points(right_x[0], right_y[0])
    for i in range(len(temp_x)):
        ax.plot(temp_x[i], temp_y[i], **plot_kwargs)
    # temp_x, temp_y = get_hand_points(left_x[0], left_y[0])
    # for i in range(len(temp_x)):
    #     ax.plot(temp_x[i], temp_y[i],)
        
    xmin, xmax = x_min_max
    plt.xlim(xmin, xmax)
    ymin, ymax = y_min_max
    plt.ylim(ymin, ymax)


@st.cache_data
def show_rep_images(
    X: npt.ArrayLike,
    n_clusters: int,
    columns: list[str],
) -> npt.ArrayLike:
    # Get clustering
    print('Start clustering')
    kmeans = cluster_frames(X, k=n_clusters)
    # joblib.dump(kmeans, f'kmeans-sign_{SIGN_NAME}-{n_clusters}.joblib')

    print('get_representative_images')
    rep_frames, rep_frame_idx = get_representative_images(
        X,
        kmeans,
        frame_subset_mask=None,
    )

    # 
    data_df = pd.DataFrame(data=rep_frames, columns=columns)


    print('Create plots')
    base_size = 3
    fig = plt.figure(
        figsize=(5*base_size, (n_clusters//5)*base_size),
    )

    for j in range(n_clusters):
        ax = fig.add_subplot(n_clusters//5, 5, j+1)
        frame = data_df.iloc[[j]]
        plot_handshape(frame, ax=ax)
    st.pyplot(fig)
    return rep_frame_idx



def display_choice(
    _frame_data: pd.DataFrame, # Not cached
    frame_idx: int,
    _col, # Not cached
):
    fig, ax = plt.subplots()
    plot_handshape(_frame_data.iloc[[frame_idx]], ax=ax)
    with _col:
        st.pyplot(fig)




def write_labels_to_file():
    pass
    

def selection_to_image(_container, label):
    try:
        image = load_selection_images()[label]
        _container.image(image, f'{label=}')
    except:
        _container.write(f'{label=}')

def create_form():
# Read in (most recent) file with same sign name & populate selection value 
    # if frame already defined
    # csvs = glob(f'label_all*{SIGN_NAME}*.csv')
    prev_signs = dict()
    # if csvs:
    #     # TODO: Decide how to handle multiple CSVs for a sign
    #     df_prev_signs = pd.read_csv(csvs[0], index_col=False,)
    #     prev_signs = pd.Series(
    #         df_prev_signs.handshape.values,
    #         index=df_prev_signs.frame_id,
    #     ).to_dict()

    form = st.form('my_form', clear_on_submit=True)
    with form:
        for i,frame_idx in enumerate(frame_index):
            col1, col2, col3 = st.columns(3)
            
            col1.write(
                f'### #{i:03}'
                f' - Frame: {data_full.iloc[frame_idx]["frame"]:_}'
            )
            col1.write(f'Phrase: `{data_full.iloc[frame_idx,-1]}`')
            
            display_choice(
                _frame_data=data_full,
                frame_idx=frame_idx,
                _col=col1,
            )
            # col1.write(f'')
            col2.selectbox(
                'handshape',
                selection_list,
                index=selection_list.index(
                    prev_signs.get(frame_idx, NO_SELECTION_STR)
                ),
                key=frame_idx,
            )
            col3.write(f'{frame_idx=}')

        submitted = st.form_submit_button(
            'Save results',
            on_click=write_labels_to_file
        )


### MAIN
preform = st.form('labeling_preset')

with preform:
    N_CLUSTERS = st.number_input(
        min_value=5,
        value=10,
        label='clusters',
        step=5,
    )

    submitted_preform = st.form_submit_button('Cluster Frames')

if submitted_preform:
    frame_index = show_rep_images(
        data.values,
        n_clusters=N_CLUSTERS,
        columns=list(data.columns),
    )

    results_container = st.container()
    create_form()