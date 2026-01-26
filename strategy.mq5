#property copyright "Gildo Marrove"
#property link      https://www.primetechmall.com
#property version   "1.11"
#property strict

#include <Trade/Trade.mqh>
CTrade trade;

// ================= INPUTS =================
input ENUM_TIMEFRAMES HTF_TF = PERIOD_W1;
input int EMA_Period = 25;
input int ATR_Period = 18;

input double MinFVGPowerPct = 0.01;
input int    FVGHistoryNbr  = 15;  // Number of FVGs to track (replaces hardcoded 10 and 50)

input double SL_ATR_Mult = 2.0;
input double TP_ATR_Mult = 19.0;

input bool   UseTrailing   = true;
input double TrailATRMult  = 2.0;

input bool   UseBreakEvenMove = false;  // Move SL to entry at X ATR profit
input double BreakEvenATRMult = 1.0;    // ATR multiple to trigger break-even move

input bool   UsePartialTP = false;      // Close N trades at X ATR profit
input double PartialTPATRMult = 1.0;    // ATR multiple to trigger partial close
input int    PartialCloseTradesCount = 1; // Number of trades to close at partial TP

bool   UseFixedLot = true;
double FixedLot    = 0.10;
double RiskPercent = 1.0;

input double PerTradeLot = 0.01;  // Lot size per trade
input int    TradeSplitCount = 1; // Number of trades to open at once

input int LiquidityLookback = 20;

input bool RequireBiasAlignment  = true;

input int    MaxDailyTrades = 10;  // Maximum trades per day
input bool   UseVolumeCheck = false;  // Enable volume check
input double VolumeMultiplier = 5;  // Volume multiplier for marketOK check
input bool   HoldUntilOpposite = false;  // Close on opposite BOS/CHoCH
input bool   AllowIntracandleChecks = true;  // If true: check price every tick for FVG touch
int    MagicNumber = 20260120;  // Magic number to track EA positions

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
   double entryPrice;
   double takeProfit;
   bool breakEvenDone;
   bool partialDone;
};

// ================= GLOBALS =================
FVG fvgList[];
int fvgCount = 0;

int dailyTradesCount = 0;
datetime lastTradeDate = 0;

bool isBOS = false;
bool isCHOCH = false;

bool lastBullFvg = false;
bool lastBearFvg = false;

TradeState tradeState;
MqlRates rates[];

datetime lastBarTime = 0;

int htfEmaHandle = INVALID_HANDLE;
int atrHandle    = INVALID_HANDLE;

// ================= INIT =================
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   htfEmaHandle = iMA(_Symbol, HTF_TF, EMA_Period, 0, MODE_EMA,
PRICE_CLOSE);
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
   int total = PositionsTotal();
   for(int i=0; i<total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      return true;
   }
   return false;
}

int GetPositionTickets(bool isLong, ulong &tickets[])
{
   ArrayResize(tickets, 0);
   int total = PositionsTotal();
   for(int i=0; i<total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      bool posLong = PositionGetInteger(POSITION_TYPE) ==
POSITION_TYPE_BUY;
      if(posLong != isLong) continue;
      int newSize = ArraySize(tickets) + 1;
      ArrayResize(tickets, newSize);
      tickets[newSize - 1] = ticket;
   }
   return ArraySize(tickets);
}

double NormalizeLotToStep(double lots)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0) return lots;
   double steps = lots / step;
   double roundedSteps = MathFloor(steps + 0.5);
   double normLots = roundedSteps * step;
   if(normLots < minLot) normLots = minLot;
   if(normLots > maxLot) normLots = maxLot;
   return NormalizeDouble(normLots, 2);
}

double AdjustStopForStopLevel(double desiredSL, bool isLong)
{
   double stopLevel = SymbolInfoInteger(_Symbol,
SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   if(stopLevel <= 0) return desiredSL;
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick) || tick.bid <= 0 || tick.ask <= 0)
      return 0;
   double bid = tick.bid;
   double ask = tick.ask;
   double curPrice = isLong ? bid : ask;
   if(isLong)
   {
      if(curPrice - desiredSL < stopLevel)
         desiredSL = curPrice - stopLevel;
   }
   else
   {
      if(desiredSL - curPrice < stopLevel)
         desiredSL = curPrice + stopLevel;
   }
   return desiredSL;
}

void UpdateStopsForPositions(bool isLong, double newSL)
{
   ulong tickets[];
   int count = GetPositionTickets(isLong, tickets);
   if(count == 0) return;
   for(int i=0; i<count; i++)
   {
      if(!PositionSelectByTicket(tickets[i])) continue;
      double curSL = PositionGetDouble(POSITION_SL);
      double curTP = PositionGetDouble(POSITION_TP);
      if(curSL != 0)
      {
         if(isLong && newSL <= curSL) continue;
         if(!isLong && newSL >= curSL) continue;
      }
      if(!trade.PositionModify(tickets[i], newSL, curTP))
      {
         Print("? Failed to modify SL. Ticket: ", tickets[i],
               " | Retcode: ", trade.ResultRetcode(),
               " | ", trade.ResultRetcodeDescription());
      }
   }
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
   int copied = (int)CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, 20,
volBuf);
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

   // Match Python: use lastBullFvg/lastBearFvg toggles
   lastBullFvg = bullFVG && !lastBullFvg;
   lastBearFvg = bearFVG && !lastBearFvg;

   double gapClose = rates[2].close;
   if(gapClose <= 0) return;

   if(lastBullFvg)
   {
      double gap = rates[1].low - rates[3].high;
      double power = (gap / gapClose) * 100.0;

      if(power >= MinFVGPowerPct)
      {
         if(fvgCount >= FVGHistoryNbr)
         {
            for(int i=1;i<fvgCount;i++)
               fvgList[i-1] = fvgList[i];
            fvgCount--;
         }
         fvgList[fvgCount].top     = rates[1].low;
         fvgList[fvgCount].bottom = rates[3].high;
         fvgList[fvgCount].bullish= true;
         fvgList[fvgCount].traded = false;
         fvgList[fvgCount].time   = rates[2].time;
         fvgCount++;
      }
   }

   if(lastBearFvg)
   {
      double gap = rates[3].low - rates[1].high;
      double power = (gap / gapClose) * 100.0;

      if(power >= MinFVGPowerPct)
      {
         if(fvgCount >= FVGHistoryNbr)
         {
            for(int i=1;i<fvgCount;i++)
               fvgList[i-1] = fvgList[i];
            fvgCount--;
         }
         fvgList[fvgCount].top     = rates[3].low;
         fvgList[fvgCount].bottom = rates[1].high;
         fvgList[fvgCount].bullish= false;
         fvgList[fvgCount].traded = false;
         fvgList[fvgCount].time   = rates[2].time;
         fvgCount++;
      }
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
      if(rates[i].high > prevStructureHigh) prevStructureHigh =
rates[i].high;
      if(rates[i].low < prevStructureLow) prevStructureLow = rates[i].low;
   }

   double prevClose = rates[2].close;
   double curClose = rates[1].close;

   // BOS: crossover(close, prevStructureHigh)
   isBOS = (prevClose <= prevStructureHigh) && (curClose >
prevStructureHigh);

   // CHoCH: crossunder(close, prevStructureLow)
   isCHOCH = (prevClose >= prevStructureLow) && (curClose <
prevStructureLow);
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

      bool touch = false;
      double entry = 0.0;
      if(AllowIntracandleChecks)
      {
         // Intrabar: check current price inside FVG zone
         double currentPrice = isLong ? ask : bid;
         touch = currentPrice <= fvgList[i].top && currentPrice >=
fvgList[i].bottom;
         // Enter at FVG boundary (same as backtest intrabar option)
         entry = isLong ? fvgList[i].top : fvgList[i].bottom;
      }
      else
      {
         // Bar-close: use last closed bar overlap
         touch = rates[1].low <= fvgList[i].top &&
               rates[1].high >= fvgList[i].bottom;
         entry = isLong ? ask : bid;
      }
      if(!touch) continue;

      double slDist = atrNorm * SL_ATR_Mult * _Point;
      double tpDist = atrNorm * TP_ATR_Mult * _Point;

      double sl = isLong ? entry - slDist : entry + slDist;
      double tp = isLong ? entry + tpDist : entry - tpDist;

      double perTradeLot = NormalizeLotToStep(PerTradeLot);
      int tradesToOpen = TradeSplitCount;
      if(perTradeLot <= 0 || tradesToOpen <= 0)
      {
         double lot = GetLot(slDist);
         perTradeLot = NormalizeLotToStep(lot);
         tradesToOpen = 1;
      }
      if(perTradeLot <= 0) continue;

      int opened = 0;
      for(int t=0; t<tradesToOpen; t++)
      {
         double orderPrice = 0.0; // market price
         bool ok = isLong
            ? trade.Buy(perTradeLot,_Symbol,orderPrice,sl,tp)
            : trade.Sell(perTradeLot,_Symbol,orderPrice,sl,tp);
         if(ok) opened++;
         else
         {
            Print("? Order failed. Retcode: ", trade.ResultRetcode(),
                  " | ", trade.ResultRetcodeDescription());
         }
      }

      if(opened > 0)
      {
         fvgList[i].traded = true;
         tradeState.hasPosition = true;
         tradeState.isLong = isLong;
         tradeState.entryATR = atr;
         tradeState.entryPrice = entry;
         tradeState.takeProfit = tp;
         tradeState.breakEvenDone = false;
         tradeState.partialDone = false;
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

   bool isLong = tradeState.isLong;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double curPrice = isLong ? bid : ask;

   // Break-even move
   if(UseBreakEvenMove && !tradeState.breakEvenDone &&
tradeState.entryATR > 0)
   {
      double beTrigger = isLong
         ? tradeState.entryPrice + tradeState.entryATR * BreakEvenATRMult
         : tradeState.entryPrice - tradeState.entryATR * BreakEvenATRMult;
      bool hit = isLong ? (curPrice >= beTrigger) : (curPrice <=
beTrigger);
      if(hit)
      {
         double beSL = AdjustStopForStopLevel(tradeState.entryPrice,
isLong);
         beSL = NormalizeDouble(beSL, _Digits);
         UpdateStopsForPositions(isLong, beSL);
         tradeState.breakEvenDone = true;
      }
   }

   // Partial close by number of trades
   if(UsePartialTP && !tradeState.partialDone && tradeState.entryATR > 0)
   {
      double partialTrigger = isLong
         ? tradeState.entryPrice + tradeState.entryATR * PartialTPATRMult
         : tradeState.entryPrice - tradeState.entryATR * PartialTPATRMult;
      bool hit = isLong ? (curPrice >= partialTrigger) : (curPrice <=
partialTrigger);
      if(hit)
      {
         ulong tickets[];
         int count = GetPositionTickets(isLong, tickets);
         int toClose = MathMin(PartialCloseTradesCount, count);
         int closed = 0;
         for(int i=0; i<toClose; i++)
         {
            if(trade.PositionClose(tickets[i]))
               closed++;
         }
         if(closed > 0)
            tradeState.partialDone = true;
      }
   }

   if(!UseTrailing) return;

   double price = isLong ? rates[0].high : rates[0].low;
   double newSL = isLong
      ? price - tradeState.entryATR * TrailATRMult
      : price + tradeState.entryATR * TrailATRMult;

   newSL = AdjustStopForStopLevel(newSL, isLong);

   newSL = NormalizeDouble(newSL,_Digits);
   UpdateStopsForPositions(isLong, newSL);
}

// ================= ON TICK =================
void OnTick()
{
if(CopyRates(_Symbol,PERIOD_CURRENT,0,LiquidityLookback+25,rates) <
LiquidityLookback+25)
      return;

   if(IsNewBar())
   {
      DetectFVG();
      CalcBOSandCHOCH();
      if(!AllowIntracandleChecks)
         TryEntry();
   }

   if(AllowIntracandleChecks)
      TryEntry();

   // Update position state (check if position still exists)
   if(!HasPosition())
   {
      tradeState.hasPosition = false;
      tradeState.breakEvenDone = false;
      tradeState.partialDone = false;
   }
   else
   {
      tradeState.hasPosition = true;
      // Update isLong if position exists
      ulong tickets[];
      int countLong = GetPositionTickets(true, tickets);
      int countShort = GetPositionTickets(false, tickets);
      if(countLong > 0) tradeState.isLong = true;
      else if(countShort > 0) tradeState.isLong = false;
   } 
   ManageTrade();
}