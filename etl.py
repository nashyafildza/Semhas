import pandas as pd
import gspread
import os
import json
from google.oauth2 import service_account
from google.cloud import bigquery
from dotenv import load_dotenv
import hashlib
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ================= CONFIG =================
PROJECT_ID = "damiu-nasqua-488213"
DATASET = "datamart_damiu_nasqua"
SPREADSHEET_ID = "1NjQf8_y2Ek7r313P3451270JXKt0ugMrMok0MbtqeIU"
WORKSHEET_NAME = "Data Input"

# ================= EXTRACT =================
def get_credentials():
    if "GCP_CREDENTIALS" not in os.environ:
        raise ValueError("❌ ENV GCP_CREDENTIALS tidak ditemukan")

    creds_dict = json.loads(os.environ["GCP_CREDENTIALS"])

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/cloud-platform"
    ]

    return service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=scopes
    )

def extract():
    creds = get_credentials()
    client = gspread.authorize(creds)
    worksheet = client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    logger.info(f"Reading sheet: {worksheet.title}")
    data = worksheet.get_all_records()

    if not data:
        raise ValueError("❌ Data kosong di sheet")

    df = pd.DataFrame(data)

    # ================= VALIDASI KOLOM =================
    expected_cols = [
        "Tanggal",
        "Nama Pelanggan",
        "Jenis Pelanggan",
        "Jenis Layanan",
        "Jumlah Galon",
        "Stok Terpakai (Liter)",
        "Harga Satuan",
        "Total Pembayaran",
        "Total Modal",
        "Laba Total"
    ]

    df.columns = df.columns.str.strip()

    column_mapping = {
    "Pembayaran": "Total Pembayaran",
    "Total Pembayaran": "Total Pembayaran",
    "Modal": "Total Modal",
    "Laba": "Laba Total"
    }

    df = df.rename(columns=column_mapping)
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Kolom tidak lengkap: {missing_cols}")

    return df

def validate(df):
    df = df.dropna(subset=["Tanggal", "Nama Pelanggan"])
    df = df.drop_duplicates()
    return df

# ================= TRANSFORM =================
def clean_currency(val):
    try:
        if pd.isna(val):
            return 0

        val = str(val).lower()
        val = val.replace("rp", "").replace(".", "").replace(",", "").strip()

        if val == "":
            return 0

        return int(val)

    except Exception:
        logger.error(f"Invalid currency: {val}")
        raise

def transform(df):

  # ================= NORMALISASI =================
    df['Nama Pelanggan'] = df['Nama Pelanggan'].astype(str).str.strip().str.lower()
    df['Jenis Pelanggan'] = df['Jenis Pelanggan'].astype(str).str.strip().str.lower()
    df['Jenis Layanan'] = df['Jenis Layanan'].astype(str).str.strip().str.lower()

    df['Jenis Pelanggan'] = df['Jenis Pelanggan'].replace('', 'unknown')

    df = df.rename(columns={
        "Tanggal": "tanggal",
        "Nama Pelanggan": "nama_pelanggan",
        "Jenis Pelanggan": "jenis_pelanggan",
        "Jenis Layanan": "jenis_layanan",
        "Jumlah Galon": "jumlah_galon",
        "Stok Terpakai (Liter)": "liter_terpakai",
        "Harga Satuan": "harga_satuan",
        "Total Pembayaran": "total_pendapatan",
        "Total Modal": "total_modal",
        "Laba Total": "total_laba"
    })

    df = df.loc[:, ~df.columns.duplicated()]
    if df.columns.duplicated().any():
        raise ValueError("❌ Duplicate column detected after rename")

    df['tanggal'] = pd.to_datetime(
        df['tanggal'], 
        dayfirst=True,
        format="%d/%m/%y",
        errors='raise'
        )

    # ================= DIM WAKTU =================
    dim_waktu = df[['tanggal']].drop_duplicates().copy()
    dim_waktu['id_waktu'] = dim_waktu['tanggal'].dt.strftime('%Y%m%d').astype(int)
    dim_waktu = dim_waktu.sort_values(by='id_waktu')

    dim_waktu['hari'] = dim_waktu['tanggal'].dt.day
    dim_waktu['nama_hari'] = dim_waktu['tanggal'].dt.day_name()
    dim_waktu['bulan'] = dim_waktu['tanggal'].dt.month
    dim_waktu['nama_bulan'] = dim_waktu['tanggal'].dt.month_name()
    dim_waktu['tahun'] = dim_waktu['tanggal'].dt.year

    dim_waktu = dim_waktu[[
    'id_waktu',
    'tanggal',
    'hari',
    'nama_hari',
    'bulan',
    'nama_bulan',
    'tahun'
    ]]

    # ================= DIM PELANGGAN =================
    dim_pelanggan = df[['nama_pelanggan', 'jenis_pelanggan']].drop_duplicates().copy()

    dim_pelanggan['id_pelanggan'] = dim_pelanggan.apply(
        lambda r: "PLG_" + hashlib.md5(
            f"{str(r['nama_pelanggan']).strip().lower()}_{str(r['jenis_pelanggan']).strip().lower()}".encode()
            ).hexdigest()[:8],
        axis=1
    )
    dim_pelanggan = dim_pelanggan[[
    'id_pelanggan',
    'nama_pelanggan',
    'jenis_pelanggan'
    ]]

    # ================= DIM LAYANAN =================
    dim_layanan = df[['jenis_layanan']].drop_duplicates().copy()

    dim_layanan['id_layanan'] = dim_layanan['jenis_layanan'].apply(
        lambda x: f"LYN_{hashlib.md5(x.encode()).hexdigest()[:8]}"
    )
    dim_layanan = dim_layanan[[
    'id_layanan',
    'jenis_layanan'
    ]]

    # ================= FACT =================
    df = df.merge(dim_pelanggan, on=["nama_pelanggan", "jenis_pelanggan"])
    df = df.merge(dim_layanan, on="jenis_layanan")

    df['id_waktu'] = df['tanggal'].dt.strftime('%Y%m%d').astype(int)

    df = df.sort_values(by=["tanggal", "nama_pelanggan"])

    df["urutan"] = df.groupby(
        ["tanggal", "nama_pelanggan"]
        ).cumcount()

    df['id_transaksi'] = df.apply(
        lambda r: hashlib.md5(
            f"{r['tanggal']}_{r['nama_pelanggan']}_{r['jenis_layanan']}_{r['urutan']}".encode()
            ).hexdigest(),
        axis=1
        )

    money_cols = [
    'harga_satuan',
    'total_pendapatan',
    'total_modal',
    'total_laba'
    ]

    for col in money_cols:
        df[col] = df[col].apply(clean_currency)

    fact = df[[
        'id_transaksi',
        'id_waktu',
        'id_pelanggan',
        'id_layanan',
        'jumlah_galon',
        'liter_terpakai',
        'harga_satuan',
        'total_pendapatan',
        'total_modal',
        'total_laba'
    ]]

    logger.info(f"Columns: {list(df.columns)}")

    logger.info(f"NULL count fact:\n{fact.isnull().sum()}")
    logger.info(f"Duplicate transaksi: {fact['id_transaksi'].duplicated().sum()}")
# ================= ENFORCE =================
    if fact.isnull().sum().sum() > 0: raise ValueError("❌ Ada NULL di fact table")
    assert fact['id_transaksi'].is_unique, "❌ Duplicate transaksi!"
    assert fact['id_pelanggan'].isin(dim_pelanggan['id_pelanggan']).all()
    assert fact['id_layanan'].isin(dim_layanan['id_layanan']).all()
    assert dim_waktu['id_waktu'].is_unique, "❌ Duplicate id_waktu"

    return fact, dim_waktu, dim_pelanggan, dim_layanan

# ================= LOAD =================
def load_table(df, table_name, client):
    try:
        table_id = f"{PROJECT_ID}.{DATASET}.{table_name}"

        client.load_table_from_dataframe(
            df,
            table_id,
            job_config=bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE"
            )
        ).result()

        logger.info(f"Loaded {table_name} ({len(df)} rows)")

    except Exception as e:
        logger.error(f"LOAD FAILED: {table_name} - {e}")
        raise

# ================= RUN =================
def run_etl():
    try:
        credentials = get_credentials()

        client = bigquery.Client(
            credentials=credentials,
            project=PROJECT_ID
        )
        logger.info("=== ETL START ===")
        # ================= EXTRACT =================
        df = extract()
        logger.info(f"Jumlah baris: {len(df)}")

        # ================= VALIDATE =================
        df = validate(df)
        logger.info(f"Jumlah Baris Setelah Validate: {len(df)}")

        # ================= TRANSFORM =================
        fact, dim_waktu, dim_pelanggan, dim_layanan = transform(df)

        logger.info(f"Fact rows: {len(fact)}")
        logger.info(f"Dim pelanggan: {len(dim_pelanggan)}")
        logger.info(f"Dim layanan: {len(dim_layanan)}")
        logger.info(f"Dim waktu: {len(dim_waktu)}")
        logger.info(fact.head())
        logger.info(f"JUMLAH DATA: {len(df)}")
        # ================= LOAD =================
        load_table(dim_waktu, "dim_waktu", client)
        load_table(dim_pelanggan, "dim_pelanggan", client)
        load_table(dim_layanan, "dim_layanan", client)
        load_table(fact, "fact_penjualan", client)

        logger.info("=== ETL SUCCESS ===")
    except Exception as e:
        logger.error(f"ETL FAILED: {e}", exc_info=True)
        raise
    
if __name__ == "__main__":
    try:
        run_etl()
    except Exception:
        exit(1)  #penting untuk GitHub Actions