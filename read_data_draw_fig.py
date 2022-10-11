import pandas as pd
import matplotlib.pyplot as plt


def draw_infections(file_name, save_path):
    df = pd.read_excel(file_name, index_col=0)
    df = df.fillna(0)
    plt.rcParams['font.size'] = 32
    plt.rcParams['font.sans-serif'] = ['simhei']
    df.plot(figsize=(32, 18), linewidth=2.0)
    bwith = 2
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=3, length=10)
    plt.ylabel('感染人数')
    plt.savefig(save_path, bbox_inches='tight', dpi=1200)


# draw_infections('infect_light.xlsx', 'infect_light_all.pdf')
# draw_infections('infect_light1.xlsx', 'infect_light.pdf')
# draw_infections('infect_heavy.xlsx', 'infect_heavy_all.pdf')
# draw_infections('infect_heavy1.xlsx', 'infect_heavy.pdf')


def get_df(file_path):
    df = pd.read_csv(file_path, index_col=0)
    y1_columns = [column for column in df.columns if 'y1' in column]
    y2_columns = [column for column in df.columns if 'y2' in column]
    y3_columns = [column for column in df.columns if 'y3' in column]
    y4_columns = [column for column in df.columns if 'y4' in column]
    df_y1 = df[y1_columns]
    df_y1.columns = [column.strip('y1') for column in df_y1.columns]
    df_y2 = df[y2_columns]
    df_y2.columns = [column.strip('y2') for column in df_y2.columns]
    df_y3 = df[y3_columns]
    df_y3.columns = [column.strip('y3') for column in df_y3.columns]
    df_y4 = df[y4_columns]
    df_y4.columns = [column.strip('y4') for column in df_y4.columns]


# df_y1, df_y2, df_y3, df_y4 = get_df('vegetables.csv')


def draw_vegetables(df, file_name, save_path):
    plt.rcParams['font.size'] = 32
    plt.rcParams['font.sans-serif'] = ['simhei']
    df.plot(figsize=(32, 18), linewidth=2.0)
    bwith = 2
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=3, length=10)
    plt.ylabel(file_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=1200)


# draw_vegetables(df_y1, '接收/自采蔬菜包数量','receive_number.pdf')
# draw_vegetables(df_y2, '接收/自采蔬菜包吨数','receive_tons.pdf')
# draw_vegetables(df_y3, '发放蔬菜包数量','give_out_number.pdf')
# draw_vegetables(df_y4, '发放蔬菜包吨数','give_out_tons.pdf')