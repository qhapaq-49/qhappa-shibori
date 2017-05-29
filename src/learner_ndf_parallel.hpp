#ifndef APERY_LEARNER_HPP
#define APERY_LEARNER_HPP

#include "position.hpp"
#include "thread.hpp"
#include "evaluate.hpp"

#if defined LEARN

#if 0
#define PRINT_PV(x) x
#else
#define PRINT_PV(x)
#endif


//see http://qiita.com/skitaoka/items/e6afbe238cd69c899b2a for example
struct AdadeltaParams{
	float r;
	float s;
	float v;
};

struct AdadeltaEvaluater{
	static float  adamgamma;
	static float  adamepsilon;

	static void init (float gamma,float epsilon) {
		adamgamma=gamma;
		adamepsilon=epsilon;
	} // float 型とかだと規格的に 0 は保証されなかった気がするが実用上問題ないだろう。

	static void onestep(AdadeltaParams &ada, float &inputgrad) {
		//adadelta
		ada.r = adamgamma*ada.r+(1-adamgamma)*inputgrad*inputgrad;
		ada.v = inputgrad*(sqrt(ada.s)+adamepsilon)/(sqrt(ada.r)+adamepsilon);
		ada.s = adamgamma*ada.s+(1-adamgamma)*ada.v*ada.v;
		inputgrad = ada.v;
	}
};

struct Gradientmaker{
	static int moveEdge;
	static int moveSlope;
	static float stepMax;
	static bool cutMinor;
	static float getReliability(int cnt) {
		if(cnt<moveEdge-moveSlope) {return 0.0;}
		if(cnt>moveEdge-moveSlope && cnt<moveEdge+moveSlope) {return 0.5+0.5*(cnt-moveEdge)/moveSlope;}

		return 1.0;
	}

	static void makeGradientAdadelta(int cnt, float &inputgrad, int defscore, AdadeltaParams &ada, bool &nzero) {
		if(cutMinor && defscore==0) {
			inputgrad=0; ada.v=0; return;
		}
		if(cnt>moveEdge-moveSlope) {
			nzero=true;
		}
		inputgrad=inputgrad*stepMax*getReliability(cnt)/(cnt+1);
		AdadeltaEvaluater::onestep(ada,inputgrad);
	}
};

struct RawEvaluater {
	int kpp_rawcnt[SquareNum][fe_end][fe_end];
	int kkp_rawcnt[SquareNum][SquareNum][fe_end];
	int kk_rawcnt[SquareNum][SquareNum];

	std::array<float, 2> kpp_raw[SquareNum][fe_end][fe_end];
	std::array<float, 2> kkp_raw[SquareNum][SquareNum][fe_end];
	std::array<float, 2> kk_raw[SquareNum][SquareNum];


	void incParam(const Position& pos, const std::array<double, 2>& dinc) {
		const Square sq_bk = pos.kingSquare(Black);
		const Square sq_wk = pos.kingSquare(White);
		const int* list0 = pos.cplist0();
		const int* list1 = pos.cplist1();
		//FVScale=32 is it really needed?
		const std::array<float, 2> f = {{static_cast<float>(dinc[0]/FVScale ), static_cast<float>(dinc[1]/FVScale )}};

		kk_raw[sq_bk][sq_wk] += f;
		kk_rawcnt[sq_bk][sq_wk] += 1;

		//mirror
		/*
		   kk_raw[inverseFile(sq_bk)][inverseFile(sq_wk)] += f;
		   kk_rawcnt[inverseFile(sq_bk)][inverseFile(sq_wk)] += 1;
		   */
		for (int i = 0; i < pos.nlist(); ++i) {
			const int k0 = list0[i];
			const int k1 = list1[i];
			for (int j = 0; j < i; ++j) {
				const int l0 = list0[j];
				const int l1 = list1[j];
				kpp_raw[sq_bk         ][k0][l0] += f;
				kpp_raw[inverse(sq_wk)][k1][l1][0] -= f[0];
				kpp_raw[inverse(sq_wk)][k1][l1][1] += f[1];
				kpp_rawcnt[sq_bk         ][k0][l0] += 1;
				kpp_rawcnt[inverse(sq_wk)][k1][l1] += 1;

				//mirror
				/*
				   kpp_raw[inverseFile(sq_bk)         ][inverseFileIndexIfOnBoard(k0)][inverseFileIndexIfOnBoard(l0)] += f;
				   kpp_raw[inverseFile(inverse(sq_wk))][inverseFileIndexIfOnBoard(k1)][inverseFileIndexIfOnBoard(l1)][0] -= f[0];
				   kpp_raw[inverseFile(inverse(sq_wk))][inverseFileIndexIfOnBoard(k1)][inverseFileIndexIfOnBoard(l1)][1] += f[1];
				   kpp_rawcnt[inverseFile(sq_bk         )][inverseFileIndexIfOnBoard(k0)][inverseFileIndexIfOnBoard(l0)] += 1;
				   kpp_rawcnt[inverse(sq_wk)][inverseFileIndexIfOnBoard(k1)][inverseFileIndexIfOnBoard(l1)] += 1;
				   */
			}
			kkp_raw[sq_bk][sq_wk][k0] += f;
			kkp_rawcnt[sq_bk][sq_wk][k0] += 1;
			//mirror
			/*
			   kkp_raw[inverseFile(sq_bk)][inverseFile(sq_wk)][inverseFileIndexIfOnBoard(k0)] += f;
			   kkp_rawcnt[inverseFile(sq_bk)][inverseFile(sq_wk)][inverseFileIndexIfOnBoard(k0)] += 1;
			   */
		}
	}

	void clear() {
		int kppsize=SquareNum*fe_end*fe_end;
		int kkpsize=SquareNum*SquareNum*fe_end;
		int kksize=SquareNum*SquareNum;
		memset(kpp_rawcnt, 0, kppsize*sizeof(int));
		memset(kkp_rawcnt, 0, kkpsize*sizeof(int));
		memset(kk_rawcnt, 0, kksize*sizeof(int));
		memset(kpp_raw, 0, kppsize*sizeof(std::array<float, 2>));
		memset(kkp_raw, 0, kkpsize*sizeof(std::array<float, 2>));
		memset(kk_raw, 0, kksize*sizeof(std::array<float, 2>));
	} // float 型とかだと規格的に 0 は保証されなかった気がするが実用上問題ないだろう。

	void mirrorParams() {
#if defined _OPENMP
#pragma omp parallel
#endif

		// KKP
		{
#ifdef _OPENMP
#pragma omp for
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int ksq1 = 0; ksq1 < SquareNum; ++ksq1) {
					for (int i = 0; i < fe_end; ++i) {
						int tksq= static_cast<int>(inverseFile(static_cast<Square>(ksq)));
						int tksq1 =static_cast<int>(inverseFile(static_cast<Square>(ksq1)));
						if( ksq< tksq) {
							continue;
						}
						if(ksq == tksq && ksq1<tksq1) {
							continue;
						}
						if(ksq == tksq && ksq1 == tksq1 && i < inverseFileIndexIfOnBoard(i)) {
							continue;
						}
						kkp_raw[ksq][ksq1][i][0]=kkp_raw[tksq][tksq1][inverseFileIndexIfOnBoard(i)][0] = kkp_raw[ksq][ksq1][i][0]+kkp_raw[tksq][tksq1][inverseFileIndexIfOnBoard(i)][0];

						kkp_raw[ksq][ksq1][i][1]=kkp_raw[tksq][tksq1][inverseFileIndexIfOnBoard(i)][1] = kkp_raw[ksq][ksq1][i][1]+kkp_raw[tksq][tksq1][inverseFileIndexIfOnBoard(i)][1];

						kkp_rawcnt[ksq][ksq1][i]=kkp_rawcnt[tksq][tksq1][inverseFileIndexIfOnBoard(i)] = kkp_rawcnt[ksq][ksq1][i]+kkp_rawcnt[tksq][tksq1][inverseFileIndexIfOnBoard(i)];
					}
				}
			}
		}

#if defined _OPENMP
#pragma omp parallel
#endif

		// KPP
		{
#ifdef _OPENMP
#pragma omp for
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int i = 0; i < fe_end; ++i) {
					for (int j = 0; j < fe_end; ++j) {
						int tksq= static_cast<int>(inverseFile(static_cast<Square>(ksq)));

						if( ksq<tksq) {
							continue;
						}
						if(ksq == tksq && i < inverseFileIndexIfOnBoard(i)) {
							continue;
						}
						if(ksq == tksq && i == inverseFileIndexIfOnBoard(i) && j < inverseFileIndexIfOnBoard(j)) {
							continue;
						}
						kpp_raw[ksq][i][j][0]=kpp_raw[tksq][inverseFileIndexIfOnBoard(i)][inverseFileIndexIfOnBoard(j)][0] = kpp_raw[ksq][i][j][0]+kpp_raw[tksq][inverseFileIndexIfOnBoard(i)][inverseFileIndexIfOnBoard(j)][0];
						kpp_raw[ksq][i][j][1]=kpp_raw[tksq][inverseFileIndexIfOnBoard(i)][inverseFileIndexIfOnBoard(j)][1] = kpp_raw[ksq][i][j][1]+kpp_raw[tksq][inverseFileIndexIfOnBoard(i)][inverseFileIndexIfOnBoard(j)][1];

						kpp_rawcnt[ksq][i][j]=kpp_rawcnt[tksq][inverseFileIndexIfOnBoard(i)][inverseFileIndexIfOnBoard(j)] = kpp_rawcnt[ksq][i][j]+kpp_rawcnt[tksq][inverseFileIndexIfOnBoard(i)][inverseFileIndexIfOnBoard(j)];
					}
				}
			}
		}

#if defined _OPENMP
#pragma omp parallel
#endif

		// KK
		{
#ifdef _OPENMP
#pragma omp for
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int ksq1 = 0; ksq1 < SquareNum; ++ksq1) {
					int tksq= static_cast<int>(inverseFile(static_cast<Square>(ksq)));
					int tksq1 =static_cast<int>(inverseFile(static_cast<Square>(ksq1)));
					if( ksq<tksq) {
						continue;
					}
					if(ksq == tksq && ksq1<tksq1) {
						continue;
					}
					kk_raw[ksq][ksq1][0]=kk_raw[tksq][tksq1][0] = kk_raw[ksq][ksq1][0]+kk_raw[tksq][tksq1][0];
					kk_raw[ksq][ksq1][1]=kk_raw[tksq][tksq1][1] = kk_raw[ksq][ksq1][1]+kk_raw[tksq][tksq1][1];
					kk_rawcnt[ksq][ksq1]=kk_rawcnt[tksq][tksq1] = kk_rawcnt[ksq][ksq1]+kk_rawcnt[tksq][tksq1];
					//	  kk_rawcnt[ksq][ksq1]   =0;
				}
			}
		}
		std::cout<<"mirror done"<<std::endl;
	}
	void combine(RawEvaluater &addparam) {
#if defined _OPENMP
#pragma omp parallel
#endif

		// KKP
		{
#ifdef _OPENMP
#pragma omp for
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int ksq1 = 0; ksq1 < SquareNum; ++ksq1) {
					for (int i = 0; i < fe_end; ++i) {
						kkp_raw[ksq][ksq1][i][0]+=addparam.kkp_raw[ksq][ksq1][i][0];
						kkp_raw[ksq][ksq1][i][1]+=addparam.kkp_raw[ksq][ksq1][i][1];
						kkp_rawcnt[ksq][ksq1][i]+=addparam.kkp_rawcnt[ksq][ksq1][i];
					}
				}
			}
		}

#if defined _OPENMP
#pragma omp parallel
#endif

		// KPP
		{
#ifdef _OPENMP
#pragma omp for
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int i = 0; i < fe_end; ++i) {
					for (int j = 0; j < fe_end; ++j) {
						kpp_raw[ksq][i][j][0]+=addparam.kpp_raw[ksq][i][j][0];
						kpp_raw[ksq][i][j][1]+=addparam.kpp_raw[ksq][i][j][1];
						kpp_rawcnt[ksq][i][j]+=addparam.kpp_rawcnt[ksq][i][j];
					}
				}
			}
		}

#if defined _OPENMP
#pragma omp parallel
#endif

		// KK
		{
#ifdef _OPENMP
#pragma omp for
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int ksq1 = 0; ksq1 < SquareNum; ++ksq1) {
					kk_raw[ksq][ksq1][0]=addparam.kk_raw[ksq][ksq1][0];
					kk_raw[ksq][ksq1][1]=addparam.kk_raw[ksq][ksq1][1];
					kk_rawcnt[ksq][ksq1]=addparam.kk_rawcnt[ksq][ksq1];
				}
			}
		}
		std::cout<<"combine done"<<std::endl;
	}
	void mirrorParams2() {
#if defined _OPENMP
#pragma omp parallel
#endif

		// KPP
		{
#ifdef _OPENMP
#pragma omp for
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int i = 0; i < fe_end; ++i) {
					for (int j = 0; j < fe_end; ++j) {
						if(i<j) {
							kpp_raw[ksq][i][j][0]=kpp_raw[ksq][j][i][0] = kpp_raw[ksq][i][j][0]+kpp_raw[ksq][j][i][0];
							kpp_raw[ksq][i][j][1]=kpp_raw[ksq][j][i][1] = kpp_raw[ksq][i][j][1]+kpp_raw[ksq][j][i][1];
							kpp_rawcnt[ksq][i][j]=kpp_rawcnt[ksq][j][i] = kpp_rawcnt[ksq][i][j]+kpp_rawcnt[ksq][j][i];
						}
					}
				}
			}
		}
		std::cout<<"mirror2 done"<<std::endl;
	}
	//normalize for adadelta
	void adadeltanormalize(std::array<AdadeltaParams, 2> kpp_ada[SquareNum][fe_end][fe_end], std::array<AdadeltaParams, 2> kkp_ada[SquareNum][SquareNum][fe_end],std::array<AdadeltaParams, 2> kk_ada[SquareNum][SquareNum]) {
		float norm = 0;
		int nzsize=0;
#if defined _OPENMP
#pragma omp parallel
#endif

		// KKP
		{
#ifdef _OPENMP
#pragma omp for reduction (+:norm,nzsize)
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int ksq1 = 0; ksq1 < SquareNum; ++ksq1) {
					for (int i = 0; i < fe_end; ++i) {
						bool nzero=false;
						Gradientmaker::makeGradientAdadelta(kkp_rawcnt[ksq][ksq1][i],kkp_raw[ksq][ksq1][i][0],1,kkp_ada[ksq][ksq1][i][0],nzero);
						Gradientmaker::makeGradientAdadelta(kkp_rawcnt[ksq][ksq1][i],kkp_raw[ksq][ksq1][i][1],1,kkp_ada[ksq][ksq1][i][1],nzero);
						norm+=kkp_raw[ksq][ksq1][i][0]*kkp_raw[ksq][ksq1][i][0]+kkp_raw[ksq][ksq1][i][1]*kkp_raw[ksq][ksq1][i][1];
						if(nzero) {
							nzsize+=1;
						}
					}
				}
			}
		}

#if defined _OPENMP
#pragma omp parallel
#endif

		// KPP
		{
#ifdef _OPENMP
#pragma omp for reduction (+:norm,nzsize)
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int i = 0; i < fe_end; ++i) {
					for (int j = 0; j < fe_end; ++j) {
						bool nzero=false;
						Gradientmaker::makeGradientAdadelta(kpp_rawcnt[ksq][i][j],kpp_raw[ksq][i][j][0],1,kpp_ada[ksq][i][j][0],nzero);
						Gradientmaker::makeGradientAdadelta(kpp_rawcnt[ksq][i][j],kpp_raw[ksq][i][j][1],1,kpp_ada[ksq][i][j][1],nzero);
						norm+=kpp_raw[ksq][i][j][0]*kpp_raw[ksq][i][j][0]+kpp_raw[ksq][i][j][1]*kpp_raw[ksq][i][j][1];
						if(nzero) {
							nzsize+=1;
						}
					}
				}
			}
		}

#if defined _OPENMP
#pragma omp parallel
#endif

		// KK
		{
#ifdef _OPENMP
#pragma omp for reduction (+:norm,nzsize)
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int ksq1 = 0; ksq1 < SquareNum; ++ksq1) {
					bool nzero=false;
					Gradientmaker::makeGradientAdadelta(kk_rawcnt[ksq][ksq1],kk_raw[ksq][ksq1][0],1,kk_ada[ksq][ksq1][0],nzero);
					Gradientmaker::makeGradientAdadelta(kk_rawcnt[ksq][ksq1],kk_raw[ksq][ksq1][1],1,kk_ada[ksq][ksq1][1],nzero);
					norm+=kk_raw[ksq][ksq1][0]*kk_raw[ksq][ksq1][0]+kk_raw[ksq][ksq1][1]*kk_raw[ksq][ksq1][1];
					if(nzero) {
						nzsize+=1;
					}
				}
			}
		}
		std::cout<<"dt norm is "<<norm<<std::endl;
		std::cout<<"nonzero param is "<<nzsize<<std::endl;
		nzsize=0;
	}

	void updateParamswithoutLowerDimension() {
		std::cout<<"update params..."<<std::endl;
#if defined _OPENMP
#pragma omp parallel
#endif

		// KKP
		{
#ifdef _OPENMP
#pragma omp for
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int ksq1 = 0; ksq1 < SquareNum; ++ksq1) {
					for (int i = 0; i < fe_end; ++i) {
						Evaluater::KKP[ksq][ksq1][i][0]+=kkp_raw[ksq][ksq1][i][0];
						Evaluater::KKP[ksq][ksq1][i][1]+=kkp_raw[ksq][ksq1][i][1];
						//	    kkp_rawcnt[ksq][ksq1][i]=0;
					}
				}
			}
		}

#if defined _OPENMP
#pragma omp parallel
#endif

		// KPP
		{
#ifdef _OPENMP
#pragma omp for
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int i = 0; i < fe_end; ++i) {
					for (int j = 0; j < fe_end; ++j) {
						Evaluater::KPP[ksq][i][j][0]+=kpp_raw[ksq][i][j][0];
						Evaluater::KPP[ksq][i][j][1]+=kpp_raw[ksq][i][j][1];
						//	    kpp_rawcnt[ksq][i][j]=0;
					}
				}
			}
		}

#if defined _OPENMP
#pragma omp parallel
#endif

		// KK
		{
#ifdef _OPENMP
#pragma omp for
#endif
			for (int ksq = 0; ksq < SquareNum; ++ksq) {
				for (int ksq1 = 0; ksq1 < SquareNum; ++ksq1) {
					Evaluater::KK[ksq][ksq1][0]+=kk_raw[ksq][ksq1][0];
					Evaluater::KK[ksq][ksq1][1]+=kk_raw[ksq][ksq1][1];
					//	  kk_rawcnt[ksq][ksq1]   =0;
				}
			}
		}
	}
};

struct Parse2Data {
	RawEvaluater params;
	bool finished;
	void clear() {
		params.clear();
	}

};

struct BookMoveData {
	bool useLearning; // 学習に使うかどうか
	Move move;
	int score;//score
};

//stat of learning data
struct learnstat{
	int gamecnt;
	int handcnt;
	int handcnt_check;
	int gamecnt_check;
};

class Learner {
public:
	void learn(Position& pos, std::istringstream& ssCmd) {
		//    eval_.readSynthesized(pos.searcher()->options["Eval_Dir"],"_synthesized_zero.bin");

		std::cout<<"loadfile done"<<std::endl;
		ssCmd >> noupdate; //just check diffsum
		ssCmd >> sfensFilename; //sfen file name
		ssCmd >> scoresFilename; //scores file name
		ssCmd >> sfensFilename_check; //sfen file name
		ssCmd >> scoresFilename_check; //scores file name
		ssCmd >> maxid;
		ssCmd >> blunderrate;
		ssCmd >> ndepth2;
		ssCmd >> ndepth3;
		/* todo test datas are needed!! */
		// [important] we don't optimize all of parameters because we only have small databese.
		// we only optimize parameters which appears as many times as moveEdge.
		// we use weight originates from number of appearance see getReliability()
		ssCmd >> Gradientmaker::moveEdge;
		ssCmd >> Gradientmaker::moveSlope;
		// assume param to be zero if original param is zero
		ssCmd >> Gradientmaker::cutMinor;
		ssCmd >> Gradientmaker::stepMax; //limit maximum value of dt    
		ssCmd >> minimumPlys; //we only use sfen after playing minimumPlys
		ssCmd >> slidePlys; //use future eval
		ssCmd >> cutDaburi;
		ssCmd >> mateFilter; //exclude sfen whose absolute score is higher than this value
		ssCmd >> bonacounts;
		ssCmd >> entropy;
		ssCmd >> adamalpha; //parameters for adam
		ssCmd >> adambeta; //parameters for adam
		ssCmd >> adamgamma; //parameters for adam
		ssCmd >> adamepsilon; //parameters for adam
		ssCmd >> updateitr; //output KPP
		ssCmd >> maxitr; //output KPP

		std::cout << "record_file: "
				  << "\nnoupdate: " << noupdate
				  << "\nsfensFilename: " << sfensFilename
				  << "\nscoresFilename: " << scoresFilename
				  << "\nsfensFilename: " << sfensFilename_check
				  << "\nscoresFilename: " << scoresFilename_check
				  << "\nmaxid: " << maxid
				  << "\nblunderrate: " << blunderrate
				  << "\nndepth2: " << ndepth2
				  << "\nndepth3: " << ndepth3
				  << "\nmoveEdge: " << Gradientmaker::moveEdge
				  << "\nmoveSlope: " << Gradientmaker::moveSlope
				  << "\ncutMinor: " << Gradientmaker::cutMinor
				  << "\nstepMax: "<<Gradientmaker::stepMax
				  << "\nminimumPlys: " << minimumPlys
				  << "\nslidePlys: " << slidePlys
				  << "\ncutDaburi: "<<cutDaburi
				  << "\nmateFilter: " << mateFilter
				  << "\nbonacounts: " << bonacounts
				  << "\nentropy: " << entropy
				  << "\nadamparams(alpha,beta,gamma,epsilon): " << adamalpha<<","<<adambeta<<","<<adamgamma<<","<<adamepsilon
				  << "\nupdateitr: "<<updateitr
				  << "\nmaxitr: "<<maxitr
				  << std::endl;
		maxid = std::max<size_t>(0, maxid);
		//parse2Data_.resize(maxid);
		std::vector<Searcher> searchers(maxid);
		for (auto& s : searchers) {
			s.init();
			positions_.push_back(Position(DefaultStartPositionSFEN, s.threads.mainThread(), s.thisptr));
		}

		readBook(pos,sfensFilename,scoresFilename,lstat_,false);
		readBook(pos,sfensFilename_check,scoresFilename_check,lstat_,true);
		std::cout<<"readbook end"<<std::endl;
		AdadeltaEvaluater::init(adamgamma,adamepsilon); //initialize adadelta
		std::cout<<"ada initialize end"<<std::endl;
		//    parse2Data_.resize(maxid);
		//positions_.resize(maxid);
		std::cout<<"resize end"<<std::endl;
		int kppsize=SquareNum*fe_end*fe_end;
		int kkpsize=SquareNum*SquareNum*fe_end;
		int kksize=SquareNum*SquareNum;

		memset(kpp_ada_hontai, 0, kppsize*sizeof(std::array<AdadeltaParams, 2>));
		memset(kkp_ada_hontai, 0, kkpsize*sizeof(std::array<AdadeltaParams, 2>));
		memset(kk_ada_hontai, 0, kksize*sizeof(std::array<AdadeltaParams, 2>));
		std::cout<<"memset end"<<std::endl;
		//    parse2Data_[0].params.kkp_ada= kkp_ada_hontai;
		//parse2Data_[0].params.kpp_ada= kpp_ada_hontai;
		//parse2Data_[0].params.kk_ada= kk_ada_hontai;

		for (int looper = 0; looper<maxitr; ++looper) {
			std::cout << "iteration " << looper << std::endl;
			std::cout << "parse1 start" << std::endl;
#if defined _OPENMP
#pragma omp parallel
#endif
			{
#ifdef _OPENMP
#pragma omp for
#endif
				for(int i=0;i<maxid;++i) {
					parse2Data_[i].finished=false;
					std::cout<<"id"<<i<<"start"<<std::endl;
					learnParse1(positions_[i],lstat_,parse2Data_[i],true,i);
					parse2Data_[i].finished=true;
				}
			}
			while(1) {
				bool finished=true;
				for(int i=0;i<maxid;++i) {
					if(!parse2Data_[i].finished) {
						finished=false;
					}
				}
				if(finished) {break;}
			}
#if defined _OPENMP
#pragma omp parallel
#endif

			{
#ifdef _OPENMP
#pragma omp for
#endif
				for(int i=0;i<maxid;++i) {
					parse2Data_[i].finished=false;
					std::cout<<"id"<<i<<"start"<<std::endl;
					learnParse1(positions_[i],lstat_,parse2Data_[i],false,i);
					parse2Data_[i].finished=true;
				}
			}

			//wait for all threads finished nennotame
			while(1) {
				bool finished=true;
				for(int i=0;i<maxid;++i) {
					if(!parse2Data_[i].finished) {
						finished=false;
					}
				}
				if(finished) {break;}
			}
			if(noupdate) {std::cout<<"check finished"<<std::endl;return;}
			if(!noupdate) {
				for(int i=1;i<maxid;++i) {
					parse2Data_[0].params.combine(parse2Data_[i].params);
				}
				parse2Data_[0].params.mirrorParams();
				parse2Data_[0].params.mirrorParams2();
				parse2Data_[0].params.adadeltanormalize(kpp_ada_hontai,kkp_ada_hontai,kk_ada_hontai);
				std::cout<<"normalize done"<<std::endl;
				parse2Data_[0].params.updateParamswithoutLowerDimension();
			}
			if(looper%updateitr==updateitr-1) {
				std::cout<<"update file"<<std::endl;
				eval_.writeSynthesized(pos.searcher()->options["Eval_Dir"]);			  
			}

		}
	}
private:
	void readBook(Position& pos, std::string sfenfile, std::string scorefile, learnstat &lstat, bool check) {
		std::ifstream ifssfen(sfenfile.c_str(), std::ios::binary);
		std::ifstream ifsscore(scorefile.c_str(), std::ios::binary);
		std::string onesfen; //sfen for one battle
		std::string onescore; //score for one battle
		std::set<Key> dict;
		if(check) {
			lstat.handcnt_check=0;
		}else{
			lstat.handcnt=0;
			lstat.gamecnt=0;
		}
		while(1) {
			std::getline(ifssfen,onesfen);
			std::getline(ifsscore,onescore);
			if(!ifssfen) {break;}
			//std::cout<<onesfen<<std::endl;
			//std::cout<<onescore<<std::endl;
			setLearnSfens(pos, onesfen, onescore,lstat, dict, check);
		}
		if(check) {
			std::cout<<"gamecnt_check: "<<lstat.gamecnt_check<<std::endl;
			std::cout<<"handcnt_check: "<<lstat.handcnt_check<<std::endl;
		}else{
			std::cout<<"gamecnt: "<<lstat.gamecnt<<std::endl;
			std::cout<<"handcnt: "<<lstat.handcnt<<std::endl;
		}
	}

	//input sfen and score
	void setLearnSfens(Position& pos, std::string onesfen, std::string onescore, learnstat &lstat,std::set<Key> &dict,bool check) {
		//bookmoves datum is meaningless for sfen learning however we save as apery was for some rainy day

		std::stringstream ssCmd(onesfen);
		std::stringstream ssCmdscore(onescore);
		pos.set(DefaultStartPositionSFEN, pos.searcher()->threads.mainThread());
		StateStackPtr setUpStates = StateStackPtr(new std::stack<StateInfo>());

		std::string token;
		int posscore; //score of position
		ssCmd >> token; //startpos
		ssCmd >> token; //moves
		int mincount=0;

		for(int i=0;i<slidePlys;++i) {
			ssCmdscore>>posscore; //score
			if(abs(posscore)>mateFilter) {return;}
		}

		while (ssCmd >> token) {
			ssCmdscore>>posscore; //score

			mincount++;
			const Move move = usiToMove(pos, token);

			if (move.isNone()) break;
			//not to add unusual gamedata
			if(mincount==1) {
				if(check) {
					bookMovesDatum_check.push_back(std::vector<BookMoveData>());
				}else{
					bookMovesDatum_.push_back(std::vector<BookMoveData>());
				}
				if(check) {
					lstat.gamecnt_check+=1;
				}else{
					lstat.gamecnt+=1;
				}
			}
			BookMoveData bmd;
			bmd.score = posscore;
			bmd.move = move;
			if(mincount>=minimumPlys) {
				if(abs(posscore)>mateFilter) {break;}
				//exclude zero assuming it is joseki or some err
				if ((dict.find(pos.getKey()) == std::end(dict) || !cutDaburi) && posscore !=0 ) {
					//use position only once and not in matefilter
					//	std::cout<<posscore<<","<<mincount<<std::endl;
					bmd.useLearning = true;
					dict.insert(pos.getKey());
					if(check) {
						lstat.handcnt_check+=1;
					}else{
						lstat.handcnt+=1;
					}
				}else{
					bmd.useLearning = false;
				}
			}else{
				bmd.useLearning = false;
			}
			if(check) {
				bookMovesDatum_check.back().push_back(bmd);
			}else{
				bookMovesDatum_.back().push_back(bmd);
			}
			setUpStates->push(StateInfo());
			pos.doMove(move, setUpStates->top());

		}
	}

	void learnParse1(Position& pos, const learnstat &lstat,Parse2Data& parse2Data,const bool check,const int id) {
		std::mt19937 mt = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
		std::uniform_int_distribution<int> dist(0, 100);
		int bonawindow=10;
		double diffsum=0;
		int troublehand=0;    
		//std::cout<<"clear done"<<std::endl;
		parse2Data.clear();
		//std::cout<<"clear done"<<std::endl;
		pos.searcher()->tt.clear();
		//std::cout<<"clear done"<<std::endl;
		g_evalTable.clear();
		//std::cout<<"clear done"<<std::endl;
		int loopcnt=lstat.gamecnt;
		if(check) {loopcnt=lstat.gamecnt_check;}
		int gcnt=0;
		int counter=0;

		for (int i = 0; i < loopcnt; i++) {
			if(i%maxid != id) {continue;}//separate by game
			StateStackPtr setUpStates = StateStackPtr(new std::stack<StateInfo>());
			pos.set(DefaultStartPositionSFEN, pos.searcher()->threads.mainThread());

			auto& gameMoves = bookMovesDatum_[i];
			if(check) {gameMoves=bookMovesDatum_check[i];}
			int tesu = gameMoves.size();
			int tecnt=0;
			bonakifuscore=0;
			bonabestscore=-30000;
			int prevscore = 0;
			//      std::cout<<std::endl;
			for (auto& bmd : gameMoves) {
				if (bmd.useLearning) {
					postoincparam(pos, bmd, diffsum,bonawindow, parse2Data, false, check || noupdate);

					// todo complete calc of hand fixed probability
					if(bonabestscore > bonakifuscore && prevscore-bonawindow < (pos.turn()==Black ? -bmd.score : bmd.score)) {
						troublehand+=1;
					}
					prevscore= (pos.turn()==Black ? bmd.score : -bmd.score);
					//if using bonanza purge
					int bcnt=0;
					bonabestscore=-30000;
					//might be better if one can select movetype
					//I expect nonpromotion of fu hisya kaku... is not needed in this routine because such a blunder will be excluded by search

					for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
						//if (ml.move() != bmd.move) {
						if(bcnt==bonacounts) {break;}
						bcnt+=1;
						//only use king move
						//if(ml.move().from()!=pos.kingSquare(pos.turn())) {continue;}
						//	      setUpStates->push(StateInfo());
						//pos.doMove(ml.move(), setUpStates->top());
						//postoincparam(pos, bmd, diffsum, parse2Data, true, check || noupdate);
						//pos.undoMove(ml.move());
						bool depth_2 = false;
						bool depth_3=false;
						int dnum=dist(mt);
						if(dnum<ndepth2) {depth_2=true;}
						if(dnum<ndepth3) {depth_3=true;}
						postoincparambona(pos, bmd, ml.move(), diffsum,bonawindow, parse2Data,  check || noupdate, depth_2,depth_3);
						//postoincparambona_qs(pos, bmd, ml.move(), diffsum,bonawindow, parse2Data,  check || noupdate);

						// }
					}

					//	  std::cout<<bmd.score <<","<<(pos.turn()==Black ? bonakifuscore : -bonakifuscore)<<","<< (pos.turn()==Black ? bonabestscore : -bonabestscore)<<std::endl;

				}else{
					bonabestscore=-30000;
				}

				setUpStates->push(StateInfo());
				pos.doMove(bmd.move, setUpStates->top());
				tecnt+=1;
				bonawindow = 10+256*tecnt/tesu;
			}
			//add meaningless if
			bonawindow=10;
			gcnt+=1;      
		}
		std::cout<<std::endl;
		if(check) {
			std::cout<<"diffsum_check : "<<diffsum<<std::endl;
			std::cout<<"troublehand_check"<<troublehand<<std::endl;
		}else{
			//    std::cout<<"import data done"<<std::endl;
			std::cout<<"diffsum : "<<diffsum<<std::endl;
			std::cout<<"troublehand"<<troublehand<<std::endl;
			//      print();
		}
	}

	//make position and score to gradient of params
	void postoincparambona_qs(Position& pos, BookMoveData &bmd, Move move, double &diffsum,int bonawindow, Parse2Data& parse2Data, bool noinc) {
		SearchStack ss[2];
		StateStackPtr setUpStates = StateStackPtr(new std::stack<StateInfo>());
		ss[0].staticEvalRaw.p[0][0] = ss[1].staticEvalRaw.p[0][0] = ScoreNotEvaluated;

		Color rootColor = pos.turn();
		int tscore = (rootColor==Black ? bmd.score : -bmd.score);
		RootMove rm;
		if(move!=bmd.move) {tscore -= bonawindow;}
		const Score cutscore = (move==bmd.move ? -ScoreMaxEvaluate : static_cast<Score>(tscore));
		const Score score = pos.searcher() ->direct_qsearch(pos, move, ss, cutscore, ScoreMaxEvaluate, rm);

		double dsig = 2.0*(getWinrate(score)-getWinrate(tscore))*getWinrate(score)*(1-getWinrate(score));
		if(entropy && score<10000 && score>-10000) {dsig = 2.0*(getWinrate(score)-getWinrate(tscore));} // tanuki-type lerning
		//    std::cout<<tscore<<","<<score<<","<<dsig<<std::endl;

		if(score > bonabestscore) {
			bonabestscore=score;
		}

		if(dsig>0) {
			dsig=dsig*blunderrate;
			diffsum+=blunderrate*(getWinrate(score)-getWinrate(tscore))*(getWinrate(score)-getWinrate(tscore));
			if(noinc) {return;}
		}else{
			if(move!=bmd.move || noinc) {
				return;
			}
			diffsum+=blunderrate*(getWinrate(score)-getWinrate(tscore))*(getWinrate(score)-getWinrate(tscore));
		}
		int recordPVIndex;
		for (recordPVIndex=0 ; !rm.pv_[recordPVIndex].isNone(); ++recordPVIndex) {
			setUpStates->push(StateInfo());
			pos.doMove(rm.pv_[recordPVIndex], setUpStates->top());
		}
		std::array<double, 2> dT = {{(rootColor == Black ? dsig : -dsig), dsig}};
		PRINT_PV(std::cout << ", score: " << score << ", dT: " << dT[0] << std::endl);
		dT[0] = -dT[0];
		dT[1] = (pos.turn() == rootColor ? -dT[1] : dT[1]);
		parse2Data.params.incParam(pos, dT);
		for (recordPVIndex-=1 ; recordPVIndex>=0; --recordPVIndex) {
			pos.undoMove(rm.pv_[recordPVIndex]);
		}

	}

	//make position and score to gradient of params
	void postoincparambona(Position& pos, BookMoveData &bmd, Move move, double &diffsum,int bonawindow, Parse2Data& parse2Data, bool noinc, bool depth_2, bool depth_3) {
		StateStackPtr setUpStates = StateStackPtr(new std::stack<StateInfo>());
		Color rootColor = pos.turn();
		int tscore = (rootColor==Black ? bmd.score : -bmd.score);
		//    tscore -= bonawindow;
		if(move!=bmd.move) {tscore -= bonawindow;}
		pos.searcher()->alpha = (move==bmd.move ? -ScoreMaxEvaluate: static_cast<Score>(tscore));
		pos.searcher()->beta  = ScoreMaxEvaluate; //limit maximum kanchigai rate
		if(depth_3) {
			go(pos, 3, move); //set depth to 1 (const)
		}
		else if(depth_2) {
			go(pos, 2, move); //set depth to 1 (const)
		}else{
			go(pos, 1, move); //set depth to 1 (const)
		}
		const Score score = pos.searcher()->rootMoves[0].score_;

		double dsig = 2.0*(getWinrate(score)-getWinrate(tscore))*getWinrate(score)*(1-getWinrate(score));
		if(entropy && score<10000 && score>-10000) {dsig = 2.0*(getWinrate(score)-getWinrate(tscore));} // tanuki-type lerning
		//double dsig = 2.0*(getWinrate(score)-getWinrate(tscore)); // tanuki-type lerning
		//remove somekind of mate
		//    if(score>10000 || score<-10000) {dsig=0;}
		//    std::cout<<tscore<<","<<score<<","<<dsig<<std::endl;

		if(score > bonabestscore) {
			bonabestscore=score;
		}

		if(dsig>0) {
			dsig=dsig*blunderrate;
			diffsum+=blunderrate*(getWinrate(score)-getWinrate(tscore))*(getWinrate(score)-getWinrate(tscore));
			if(noinc) {return;}
		}else{
			if(bmd.move!=move || noinc) {
				return;
			}
			diffsum+=blunderrate*(getWinrate(score)-getWinrate(tscore))*(getWinrate(score)-getWinrate(tscore));
		}

		auto& pv = pos.searcher()->rootMoves[0].pv_;
		int recordPVIndex;
		for (recordPVIndex=0 ; !pv[recordPVIndex].isNone(); ++recordPVIndex) {
			setUpStates->push(StateInfo());
			pos.doMove(pv[recordPVIndex], setUpStates->top());
		}
		std::array<double, 2> dT = {{(rootColor == Black ? dsig : -dsig), dsig}};
		PRINT_PV(std::cout << ", score: " << score << ", dT: " << dT[0] << std::endl);
		dT[0] = -dT[0];
		dT[1] = (pos.turn() == rootColor ? -dT[1] : dT[1]);
		parse2Data.params.incParam(pos, dT);
		for (recordPVIndex-=1 ; recordPVIndex>=0; --recordPVIndex) {
			pos.undoMove(pv[recordPVIndex]);
		}
	}

	//make position and score to gradient of params
	void postoincparam(Position& pos, BookMoveData &bmd, double &diffsum, int bonawindow,Parse2Data& parse2Data, bool isbona ,bool noinc) {
		SearchStack ss[2];
		ss[0].staticEvalRaw.p[0][0] = ss[1].staticEvalRaw.p[0][0] = ScoreNotEvaluated;
		Color rootColor = pos.turn();
		if(isbona) {
			if(pos.turn()==Black) {
				rootColor=White;
			}else{
				rootColor=Black;
			}
		}
		const Score score = (rootColor == pos.turn() ? evaluate(pos, ss+1) : -evaluate(pos, ss+1));
		int tscore = (rootColor==Black ? bmd.score : -bmd.score);
		if(isbona) {tscore -= bonawindow;} 

		//minimize |diff|^2=diffsum
		double dsig = 2.0*(getWinrate(score)-getWinrate(tscore))*getWinrate(score)*(1-getWinrate(score));
		if(entropy && score<10000 && score>-10000) {dsig = 2.0*(getWinrate(score)-getWinrate(tscore));} // tanuki-type lerning
		// dsig = 2.0*(getWinrate(score)-getWinrate(tscore)); // tanuki-type lerning
		//    std::cout<<tscore<<","<<score<<","<<dsig<<std::endl;
		if(!isbona) {
			bonakifuscore=score;
		}else{
			if(score > bonabestscore) {
				bonabestscore=score;
			}
		}
		if(dsig>0) {
			dsig=dsig*blunderrate;
			diffsum+=blunderrate*(getWinrate(score)-getWinrate(tscore))*(getWinrate(score)-getWinrate(tscore));
		}else{
			if(isbona) {return;}
			diffsum+=(getWinrate(score)-getWinrate(tscore))*(getWinrate(score)-getWinrate(tscore));
		}

		std::array<double, 2> dT = {{(rootColor == Black ? dsig : -dsig), dsig}};
		PRINT_PV(std::cout << ", score: " << score << ", dT: " << dT[0] << std::endl);
		dT[0] = -dT[0];
		dT[1] = (pos.turn() == rootColor ? -dT[1] : dT[1]);
		if(!noinc) {parse2Data.params.incParam(pos, dT);}    
	}

	//expect shoritu
	double getWinrate(const double x) {
		//is it really okay?
		return 1.0/(1.0+exp(-x/600));
	}

	void print() {
		for (Rank r = Rank1; r < RankNum; ++r) {
			for (File f = File9; File1 <= f; --f) {
				const Square sq = makeSquare(f, r);
				printf("%5d", Evaluater::KPP[SQ88][f_gold + SQ78][f_gold + sq][0]);
			}
			printf("\n");
		}
		printf("\n");
		fflush(stdout);
	}

	std::vector<Position> positions_;
	std::vector<std::vector<BookMoveData> > bookMovesDatum_;
	std::vector<std::vector<BookMoveData> > bookMovesDatum_check;

	Parse2Data parse2Data_[12];
	//std::vector<Parse2Data> parse2Data_;
	EvaluaterBase<std::array<std::atomic<float>, 2>,
		std::array<std::atomic<float>, 2>,
		std::array<std::atomic<float>, 2> > parse2EvalBase_;
	Evaluater eval_;

	int stepNum_;
	size_t gameNumForIteration_;

	std::string sfensFilename;
	std::string scoresFilename;
	std::string sfensFilename_check;
	std::string scoresFilename_check;
	float blunderrate;
	int minimumPlys;
	int slidePlys;
	int mateFilter;
	double adamalpha;
	double adambeta;
	double adamgamma;
	double adamepsilon;


	bool cutDaburi;
	int updateitr;
	int maxitr;
	learnstat lstat_;
	int bonacounts;
	bool noupdate;
	bool entropy;
	int bonakifuscore;
	int bonabestscore;
	int bonaprevscore;
	int maxid;
	std::array<AdadeltaParams, 2> kpp_ada_hontai[SquareNum][fe_end][fe_end];
	std::array<AdadeltaParams, 2> kkp_ada_hontai[SquareNum][SquareNum][fe_end];
	std::array<AdadeltaParams, 2> kk_ada_hontai[SquareNum][SquareNum];
	int ndepth2; //set percentage of depth2
	int ndepth3;
};

#endif

#endif // #ifndef APERY_LEARNER_HPP
