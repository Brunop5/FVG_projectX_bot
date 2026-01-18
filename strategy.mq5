#property copyright "Gildo REDACTED_USERNAME"
#property link      "https://www.primetechmall.com"
#property version   "1.11"
#property strict

#include <Trade/Trade.mqh>
CTrade trade;

// ================= INPUTS =================
input ENUM_TIMEFRAMES HTF_TF = PERIOD_H4;
input int EMA_Period = 50;
input int ATR_Period = 14;

input double MinFVGPowerPct = 0.25;
input int    FVGHistoryNbr  = 15;  // Number of FVGs to track (replaces hardcoded 10 and 50)

input double SL_ATR_Mult = 4.0;
input double TP_ATR_Mult = 20.0;

input bool   UseTrailing   = true;
input double TrailATRMult  = 6.0;

input bool   UseFixedLot = true;
input double FixedLot    = 0.10;
input double RiskPercent = 1.0;

input int LiquidityLookback = 20;

input bool RequireLiquiditySweep = false;
input bool RequireBiasAlignment  = true;
input bool DebugMode              = true;

input int    MaxDailyTrades = 3;  // Maximum trades per day
input bool   UseVolumeCheck = true;  // Enable volume check
input double VolumeMultiplier = 1.25;  // Volume multiplier for marketOK check
input bool   HoldUntilOpposite = false;  // Close on opposite BOS/CHoCH

// ================= STRUCTS =================
struct FVG
{
   double top;
   double bottom;
   bool bullish;
   bool traded;
   datetime time;
};

struct TradeState
{
   bool hasPosition;
   bool isLong;
   double entryATR;
   double takeProfit;
};

// ================= GLOBALS =================
FVG fvgList[];
int fvgCount = 0;

int dailyTradesCount = 0;
datetime lastTradeDate = 0;

bool isBOS = false;
bool isCHOCH = false;

TradeState tradeState;
MqlRates rates[];

datetime lastBarTime = 0;

int htfEmaHandle = INVALID_HANDLE;
int atrHandle    = INVALID_HANDLE;

// ================= INIT =================
int OnInit()
{
   htfEmaHandle = iMA(_Symbol, HTF_TF, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   atrHandle    = iATR(_Symbol, PERIOD_CURRENT, ATR_Period);

   if(htfEmaHandle == INVALID_HANDLE || atrHandle == INVALID_HANDLE)
      return INIT_FAILED;

   ArraySetAsSeries(rates, true);
   tradeState.hasPosition = false;

   Print("EA Initialized Successfully");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(htfEmaHandle != INVALID_HANDLE) IndicatorRelease(htfEmaHandle);
   if(atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);
}

// ================= UTIL =================
bool IsNewBar()
{
   datetime t = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(t != lastBarTime)
   {
      lastBarTime = t;
      return true;
   }
   return false;
}

bool HasPosition()
{
   return PositionSelect(_Symbol);
}

// ================= MARKET REGIME =================
bool IsMarketOK()
{
   double atrBuf[];
   ArraySetAsSeries(atrBuf, true);

   if(CopyBuffer(atrHandle, 0, 0, 20, atrBuf) < 20)
      return false;

   bool atrOK = atrBuf[1] > atrBuf[19];  // Current ATR > SMA of ATR (simplified)

   if(UseVolumeCheck == false) return atrOK;

   // Volume check
   long volBuf[];
   ArraySetAsSeries(volBuf, true);
   int copied = (int)CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, 20, volBuf);
   if(copied < 20)
      return atrOK;

   double avgVol = 0.0;
   for(int i=1; i<20; i++)
      avgVol += (double)volBuf[i];
   avgVol /= 19.0;

   bool volOK = volBuf[1] > (avgVol * VolumeMultiplier);
   return volOK && atrOK;
}

// ================= HTF BIAS =================
int GetHTFBias()
{
   double emaBuf[];
   ArraySetAsSeries(emaBuf, true);

   if(CopyBuffer(htfEmaHandle, 0, 0, 2, emaBuf) < 2)
      return 0;

   double closeHTF = iClose(_Symbol, HTF_TF, 1);

   if(closeHTF > emaBuf[1]) return 1;
   if(closeHTF < emaBuf[1]) return -1;
   return 0;
}

// ================= FVG DETECTION =================
void DetectFVG()
{
   if(ArraySize(rates) < 4) return;

   // Resize array if needed
   if(ArraySize(fvgList) < FVGHistoryNbr)
      ArrayResize(fvgList, FVGHistoryNbr);

   bool bullFVG = rates[3].high < rates[1].low;
   bool bearFVG = rates[3].low  > rates[1].high;

   double gapClose = rates[2].close;
   if(gapClose <= 0) return;

   if(bullFVG)
   {
      double gap = rates[1].low - rates[3].high;
      double power = (gap / gapClose) * 100.0;

      if(power >= MinFVGPowerPct)
      {
         fvgList[fvgCount].top     = rates[1].low;
         fvgList[fvgCount].bottom = rates[3].high;
         fvgList[fvgCount].bullish= true;
         fvgList[fvgCount].traded = false;
         fvgList[fvgCount].time   = rates[2].time;
         fvgCount++;
      }
   }

   if(bearFVG)
   {
      double gap = rates[3].low - rates[1].high;
      double power = (gap / gapClose) * 100.0;

      if(power >= MinFVGPowerPct)
      {
         fvgList[fvgCount].top     = rates[3].low;
         fvgList[fvgCount].bottom = rates[1].high;
         fvgList[fvgCount].bullish= false;
         fvgList[fvgCount].traded = false;
         fvgList[fvgCount].time   = rates[2].time;
         fvgCount++;
      }
   }

   // Limit to FVGHistoryNbr (remove oldest if exceeded)
   if(fvgCount > FVGHistoryNbr)
   {
      for(int i=1;i<fvgCount;i++)
         fvgList[i-1] = fvgList[i];
      fvgCount--;
   }
}

// ================= LOT =================
double GetLot(double slDist)
{
   if(UseFixedLot) return FixedLot;

   double risk = AccountInfoDouble(ACCOUNT_BALANCE) * RiskPercent / 100.0;
   double tickVal = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(slDist <= 0 || tickVal <= 0 || tickSize <= 0) return 0;

   double lot = risk / (slDist / tickSize * tickVal);
   return NormalizeDouble(lot, 2);
}

// ================= DAILY TRADE LIMIT =================
bool CheckDailyTradeLimit()
{
   datetime today = iTime(_Symbol, PERIOD_D1, 0);
   
   if(lastTradeDate != today)
   {
      dailyTradesCount = 0;
      lastTradeDate = today;
   }
   
   return dailyTradesCount < MaxDailyTrades;
}

// ================= BOS/CHoCH CALCULATION =================
void CalcBOSandCHOCH()
{
   if(ArraySize(rates) < 21) return;
   
   double prevStructureHigh = rates[20].high;
   double prevStructureLow = rates[20].low;
   
   for(int i=19; i>=1; i--)
   {
      if(rates[i].high > prevStructureHigh) prevStructureHigh = rates[i].high;
      if(rates[i].low < prevStructureLow) prevStructureLow = rates[i].low;
   }
   
   double prevClose = rates[2].close;
   double curClose = rates[1].close;
   
   // BOS: crossover(close, prevStructureHigh)
   isBOS = (prevClose <= prevStructureHigh) && (curClose > prevStructureHigh);
   
   // CHoCH: crossunder(close, prevStructureLow)
   isCHOCH = (prevClose >= prevStructureLow) && (curClose < prevStructureLow);
}

// ================= ENTRY =================
void TryEntry()
{
   if(HasPosition() || fvgCount == 0) return;
   if(!IsMarketOK()) return;
   if(!CheckDailyTradeLimit()) return;

   int bias = GetHTFBias();

   double atrBuf[];
   ArraySetAsSeries(atrBuf, true);
   if(CopyBuffer(atrHandle, 0, 0, 2, atrBuf) < 2) return;

   double atr = atrBuf[1];
   double atrNorm = atr / _Point;

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   for(int i=0;i<fvgCount;i++)
   {
      if(fvgList[i].traded) continue;

      bool isLong = fvgList[i].bullish;

      if(RequireBiasAlignment)
      {
         if(isLong && bias < 0) continue;
         if(!isLong && bias > 0) continue;
      }

      bool touch = rates[1].low <= fvgList[i].top &&
                   rates[1].high >= fvgList[i].bottom;
      if(!touch) continue;

      double entry = isLong ? ask : bid;
      double slDist = atrNorm * SL_ATR_Mult * _Point;
      double tpDist = atrNorm * TP_ATR_Mult * _Point;

      double sl = isLong ? entry - slDist : entry + slDist;
      double tp = isLong ? entry + tpDist : entry - tpDist;

      double lot = GetLot(slDist);
      if(lot <= 0) continue;

      bool ok = isLong
         ? trade.Buy(lot,_Symbol,entry,sl,tp)
         : trade.Sell(lot,_Symbol,entry,sl,tp);

      if(ok)
      {
         fvgList[i].traded = true;
         tradeState.hasPosition = true;
         tradeState.isLong = isLong;
         tradeState.entryATR = atr;
         tradeState.takeProfit = tp;
         dailyTradesCount++;
         lastTradeDate = iTime(_Symbol, PERIOD_D1, 0);
         return;
      }
   }
}

// ================= TRAILING =================
void ManageTrade()
{
   if(!HasPosition()) return;
   
   // Check BOS/CHoCH exits first
   if(HoldUntilOpposite)
   {
      if(tradeState.isLong && isCHOCH)
      {
         trade.PositionClose(_Symbol);
         tradeState.hasPosition = false;
         return;
      }
      
      if(!tradeState.isLong && isBOS)
      {
         trade.PositionClose(_Symbol);
         tradeState.hasPosition = false;
         return;
      }
   }
   
   if(!UseTrailing) return;

   double price = tradeState.isLong ? rates[0].high : rates[0].low;
   double newSL = tradeState.isLong
      ? price - tradeState.entryATR * TrailATRMult
      : price + tradeState.entryATR * TrailATRMult;

   newSL = NormalizeDouble(newSL,_Digits);

   double curSL = PositionGetDouble(POSITION_SL);
   if(curSL == 0) return;

   if(tradeState.isLong && newSL <= curSL) return;
   if(!tradeState.isLong && newSL >= curSL) return;

   trade.PositionModify(_Symbol,newSL,tradeState.takeProfit);
}

// ================= ON TICK =================
void OnTick()
{
   if(CopyRates(_Symbol,PERIOD_CURRENT,0,LiquidityLookback+25,rates) < LiquidityLookback+25)
      return;

   if(IsNewBar())
   {
      DetectFVG();
      CalcBOSandCHOCH();
      TryEntry();
   }
   
   // Update position state (check if position still exists)
   if(!HasPosition())
   {
      tradeState.hasPosition = false;
   }
   else if(tradeState.hasPosition)
   {
      // Update isLong if position exists
      tradeState.isLong = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
   }

   ManageTrade();
}