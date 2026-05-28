"""
data_loader.py
SPY와 VIX 일별 데이터를 Yahoo Finance에서 수집하고 전처리

[강의 연결]
- Chapter 1, 11페이지: Data pre-processing
  환경의 raw observation을 에이전트가 사용할 상태로 변환
- 이동평균선 계산, 결측치 처리 등이 여기에 해당
"""

import yfinance as yf
import pandas as pd
import numpy as np
import config


def download_raw_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Yahoo Finance에서 특정 티커의 일별 데이터를 받아온다.
    
    Args:
        ticker: 종목 코드 ("SPY" 또는 "^VIX")
        start: 시작 날짜 ("YYYY-MM-DD")
        end: 종료 날짜 ("YYYY-MM-DD")
    
    Returns:
        Date를 인덱스로 하는 DataFrame
    """
    print(f"[데이터 다운로드] {ticker}: {start} ~ {end}")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"{ticker} 데이터를 가져오지 못했습니다.")
    
    # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df


def build_dataset(start: str, end: str) -> pd.DataFrame:
    """
    SPY와 VIX 데이터를 결합하고 이동평균선 등 파생 변수를 계산한다.
    
    Returns:
        다음 컬럼을 가진 DataFrame:
        - spy_close: SPY 종가
        - vix_close: VIX 종가
        - spy_ma: SPY의 N일 이동평균
        - spy_return: SPY 일별 수익률 (학습 전 분석용)
    """
    # 1. SPY와 VIX 각각 다운로드
    spy = download_raw_data(config.TICKER_SPY, start, end)
    vix = download_raw_data(config.TICKER_VIX, start, end)
    
    # 2. 종가만 추출
    spy_close = spy["Close"].rename("spy_close")
    vix_close = vix["Close"].rename("vix_close")
    
    # 3. 두 시계열을 같은 날짜로 합치기 (둘 다 데이터가 있는 날만)
    df = pd.concat([spy_close, vix_close], axis=1).dropna()
    
    # 4. 이동평균선 계산 (Chapter 1의 pre-processing 단계)
    df["spy_ma"] = df["spy_close"].rolling(window=config.MA_PERIOD).mean()
    
    # 5. 일별 수익률 (나중에 분석용으로 사용)
    df["spy_return"] = df["spy_close"].pct_change()
    
    # 6. 이동평균 계산으로 생긴 초기 NaN 제거
    df = df.dropna().reset_index()
    
    print(f"[데이터 완성] 총 {len(df)}일치 데이터")
    print(f"  - SPY 종가 범위: {df['spy_close'].min():.2f} ~ {df['spy_close'].max():.2f}")
    print(f"  - VIX 범위: {df['vix_close'].min():.2f} ~ {df['vix_close'].max():.2f}")
    
    return df


def get_train_test_data():
    """
    학습용 데이터와 테스트용 데이터를 각각 반환한다.
    
    Returns:
        train_df, test_df: 두 개의 DataFrame
    """
    train_df = build_dataset(config.TRAIN_START, config.TRAIN_END)
    test_df = build_dataset(config.TEST_START, config.TEST_END)
    
    return train_df, test_df


# ============================================================
# 단독 실행 시 데이터 다운로드 + 간단한 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("데이터 다운로드 및 전처리 테스트")
    print("=" * 60)
    
    train_df, test_df = get_train_test_data()
    
    print("\n[학습 데이터 미리보기]")
    print(train_df.head())
    print(f"\n[테스트 데이터 미리보기]")
    print(test_df.head())
    
    # 간단한 통계 확인
    print("\n[학습 데이터 통계]")
    print(train_df[["spy_close", "vix_close", "spy_ma"]].describe())
    
    # 저장(선택사항): 매번 다운로드하기 싫으면 캐시로 사용
    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)
    print("\n[저장 완료] train_data.csv, test_data.csv")
