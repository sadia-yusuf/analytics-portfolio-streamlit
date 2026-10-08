"""One-off data prep: shrinks the raw project files into small aggregates the app can load fast."""
import pandas as pd, re, collections, sys, shutil
R = "LSE_Project_2/LSE assignment_Final/"
O = "streamlit_portfolio/data/"
which = sys.argv[1]
if which == "ar":
    ar = pd.read_csv(R+"appointments_regional.csv").drop_duplicates()
    g = ar.groupby(["appointment_month","appointment_status","hcp_type","appointment_mode","time_between_book_and_appointment"], as_index=False)["count_of_appointments"].sum()
    g.to_csv(O+"nhs_ar.csv", index=False); print(g.shape, ar.shape)
if which == "ad":
    ad = pd.read_csv(R+"actual_duration.csv")
    ad["month"] = pd.to_datetime(ad["appointment_date"]).dt.to_period("M").astype(str)
    g = ad.groupby(["month","actual_duration"], as_index=False)["count_of_appointments"].sum()
    g.to_csv(O+"nhs_ad.csv", index=False); print(g.shape)
if which == "nc":
    nc = pd.read_excel(R+"national_categories.xlsx")
    nc["month"] = pd.to_datetime(nc["appointment_date"]).dt.to_period("M").astype(str)
    g = nc.groupby(["month","service_setting","context_type","national_category"], as_index=False)["count_of_appointments"].sum()
    g.to_csv(O+"nhs_nc.csv", index=False); print(g.shape)
if which == "tw":
    tw = pd.read_csv(R+"tweets.csv"); print(tw.columns.tolist())
    col = [c for c in tw.columns if "text" in c.lower()][0]
    c = collections.Counter(h.lower() for t in tw[col].dropna() for h in re.findall(r"#\w+", t))
    pd.DataFrame(c.most_common(40), columns=["hashtag","count"]).to_csv(O+"tweet_hashtags.csv", index=False); print(c.most_common(10))
if which == "mk":
    df = pd.read_excel("LSE_Project_1/marketing_data_Analysis.xlsx", sheet_name="marketing_data")
    ad = pd.read_excel("LSE_Project_1/marketing_data_Analysis.xlsx", sheet_name="ad_data")
    ad = ad.dropna(subset=["ID"]).drop_duplicates("ID"); ad["ID"] = ad["ID"].astype(int)
    m = df.merge(ad, on="ID", how="left"); m.to_csv(O+"marketing.csv", index=False); print(m.shape)
if which == "tg":
    shutil.copy("LSE_project 3/turtle_reviews.csv", O+"turtle_reviews.csv")
if which == "bk":
    B = "Projects- Github/Banking Transaction Analysis/"
    shutil.copy(B+"data/transactions.csv", O+"transactions.csv")
    shutil.copy(B+"output/anomaly_flagged.csv", O+"anomaly_flagged.csv")
