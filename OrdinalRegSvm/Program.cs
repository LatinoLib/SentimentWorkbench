using System;
using System.Collections.Concurrent;
using System.Configuration;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Amib.Threading;
using Latino;
using Latino.Model;
using Latino.Model.Eval;
using Latino.TextMining;
using OrdinalReg.DataSource;
using TextTask;
using TextTask.Classifier;
using TextTask.DataSource;
using Action = System.Action;

namespace OrdinalReg
{
    public class TweetDataSet
    {
        public LabeledTextSource Train { get; set; }
        public LabeledTextSource Test { get; set; }
    }

    public enum CsvOutput
    {
        None,
        Metrics,
        Distances,
        DistanceProbs,
        DistanceProbsDisagr
    }

    class Program
    {

        private class TupleItem1Comparer : IComparer<Tuple<double, double>>
        {
            public static readonly TupleItem1Comparer Instance = new TupleItem1Comparer();

            private TupleItem1Comparer()
            {
            }

            public int Compare(Tuple<double, double> x, Tuple<double, double> y)
            {
                return x.Item1 < y.Item1 ? -1 : (x.Item1 > y.Item1 ? 1 : 0);
            }
        }

        public static string GetConnectionString(string connectionStringId = "default")
        {
            var conSett = ConfigurationManager.ConnectionStrings[connectionStringId];
            return conSett == null ? null : conSett.ConnectionString;
        }

        static void Main(string[] args)
        {
            // disable latino logger
            Logger.GetRootLogger().LocalOutputType = Logger.OutputType.Custom;
            Logger.GetRootLogger().LocalProgressOutputType = Logger.ProgressOutputType.Custom;

            var models = ProgramModel.SvmTwoPlaneBW1 |
                         ProgramModel.DeltaBowSpace;
           

            var dataSets = new List<TweetDataSet>
            {
                    
            };


            dataSets.AddRange(GeneratedSource.GetFromDirectory((@"..\..\GeneratedData\")));
           
            var actions = ModelActions.Validate;
            //var actions = ModelActions.Test;
            //var actions = ModelActions.Save;
            //var actions = ModelActions.Validate | ModelActions.Save | ModelActions.Test;

            Thread.CurrentThread.CurrentCulture = CultureInfo.CreateSpecificCulture("en");

            bool reportPerDataSource = false;
            var programCsvOutput = CsvOutput.Metrics;

            foreach (TweetDataSet dataSet in dataSets)
            {
                StreamWriter output = reportPerDataSource
                    ? new StreamWriter(string.Format(@"report-{0}.txt", (dataSet.Train ?? dataSet.Test).Name), false)
                    : new StreamWriter(@"report.txt", true);

                if (dataSet.Train != null)
                {
                    ModelsParams modelsParams = models.HasFlag(ProgramModel.DeltaBowSpace)
                        ? PrepareModel<SentimentDeltaBowSpace>(dataSet.Train, models, output, programCsvOutput)
                        : PrepareModel<BowSpace>(dataSet.Train, models, output, programCsvOutput);
                    modelsParams.IsBuildSvmPerfModel = actions.HasFlag(ModelActions.BuildPerf);
                    modelsParams.ExamplePredictions = null; 

                    if (actions.HasFlag(ModelActions.Validate))
                    {
                        StreamWriter csvOutput = programCsvOutput != CsvOutput.None ? new StreamWriter(@"report-validate.csv", true) : null;
                        ValidateModels(modelsParams, ExecutionMode.FoldParallelSmart, output, csvOutput);
                    }
                    if (actions.HasFlag(ModelActions.Save))
                    {
                        StreamWriter csvOutput = programCsvOutput != CsvOutput.None ? new StreamWriter(@"report-save-" + dataSet.Train.Name + ".csv", false) : null;
                        SaveModel(modelsParams, output, csvOutput);
                    }
                }

                if (dataSet.Test != null)
                {
                    if (actions.HasFlag(ModelActions.Test))
                    {
                        ModelsParams modelsParams = models.HasFlag(ProgramModel.DeltaBowSpace)
                            ? PrepareModel<SentimentDeltaBowSpace>(dataSet.Test, models, output, programCsvOutput)
                            : PrepareModel<BowSpace>(dataSet.Test, models, output, programCsvOutput);

                        StreamWriter csvOutput = programCsvOutput != CsvOutput.None ? new StreamWriter(@"report-testing.csv", true) : null;
                        TestModel(modelsParams, output, csvOutput);
                    }
                }

                output.Flush();
                output.Close();
            }
        }

        private static readonly HappinessMapper mHappinessMapper = new HappinessMapper(@"data\words_happiness.json");

        private static IEnumerable<string> GetHappinessTokens(IEnumerable<string> tokens)
        {
            foreach (string token in tokens)
            {
                yield return token;
                string hapinessTag = mHappinessMapper.Get(token);
                if (hapinessTag != null)
                {
                    yield return hapinessTag;
                }
            }
        }

        private static ModelsParams PrepareModel<T>(LabeledTextSource dataSource, ProgramModel model, StreamWriter output, CsvOutput csvOutput) where T : BowSpace, new()
        {
            var sw = new Stopwatch();
            sw.Start();

            output.WriteLine("\n*****************************************************************\n");
            output.WriteLine("Data source: {0}", dataSource.Name);
            output.WriteLine("Language: {0}", dataSource.Language);
            output.WriteLine("Data models: {0}\n", model);
            output.WriteLine("Start time: {0}", DateTime.Now);
            output.WriteLine("Loading data...");
            output.Flush();

            Console.WriteLine("\n*****************************************************************\n");
            Console.WriteLine("Data source: {0}", dataSource.Name);
            Console.WriteLine("Data models: {0}\n", model);
            Console.WriteLine("Start time: {0}", DateTime.Now);

            // load data
            dataSource.GetData();
            output.WriteLine("Loaded {0} units in {1}", dataSource.DataSize, sw.Elapsed);
            output.Flush();

            var lang = dataSource.Language;
            var wordWeightType = WordWeightType.TfIdf;
            System.Func<BowSpace> bowSpaceFunc = () =>
            {
                T bowSpace = SentimentUtils.CreateBowSpace<T>(
                    false, 2, 
                    typeof(DeltaBowSpace<SentimentLabel>).IsAssignableFrom(typeof(T)) 
                        ? WordWeightType.TermFreq 
                        : wordWeightType, 
                    lang, 5);

                //bowSpace.OnGetTokens = GetHappinessTokens;
                return bowSpace;
            };
            output.WriteLine(SentimentUtils.GetObjectReport(bowSpaceFunc(), "Logger"));
            output.Flush();

            // features 

            var featureProcessor = new TextFeatureProcessor()
                .With(new SocialMediaProcessing.NormalizeDiacriticalCharactersFeature())
                .With(new SocialMediaProcessing.UrlFeature())
                .With(new SocialMediaProcessing.MessageLengthFeature())
                .With(new SocialMediaProcessing.StockSymbolFeature())
                .With(new SocialMediaProcessing.UppercasedFeature())
                .With(new SocialMediaProcessing.TwitterUserFeature())
                .With(new SocialMediaProcessing.HashTagFeature())
                //.With(new SocialMediaProcessing.NegationFeature(lang))

                // //.With(new SocialMediaProcessing.SwearingFeature(lang))
                // //.With(new SocialMediaProcessing.PositiveWordFeature(lang))

                .With(new SocialMediaProcessing.PunctuationFeature())

                .With(new SocialMediaProcessing.LastSadEmoticonsFeature())
                .With(new SocialMediaProcessing.LastHappyEmoticonsFeature())
                .With(new SocialMediaProcessing.HappySadEmoticonsFeature())

                .With(new SocialMediaProcessing.RepetitionFeature());
            

            output.WriteLine("Feature processor config:\n{0}", featureProcessor);
            output.Flush();

            return new ModelsParams
            {
                BowSpaceFunc = bowSpaceFunc,
                Model = model,
                FeatureProcessor = featureProcessor,
                LabeledDataset = new LabeledDataset<SentimentLabel, Tweet>(dataSource.GetData<Tweet>()),
                TrainDataSource = dataSource,
                CsvOutput = csvOutput,
                    ModelFileNameFunc = modelName => string.Format("model-{0}-{1}.bin", modelName, Path.GetFileNameWithoutExtension(dataSource.Name))
            };
        }


        private static System.Func<IModel<SentimentLabel, SparseVector<double>>>[] GetModelFactory(ProgramModel model)
        {
            System.Func<IModel<SentimentLabel, SparseVector<double>>>[] modelFactory =
                Enum.GetValues(typeof(ProgramModel)).Cast<ProgramModel>()
                    .Where(pm => pm != ProgramModel.None && pm != ProgramModel.DeltaBowSpace && model.HasFlag(pm))
                    .Select<ProgramModel, System.Func<IModel<SentimentLabel, SparseVector<double>>>>(pm =>
                    {
                        switch (pm)
                        {
                            case ProgramModel.NaiveBayes:
                                return () => new BowNaiveBayesModel<SentimentLabel>();

                            case ProgramModel.NeutralZone:
                                return () => new NeutralZoneClassifier
                                {
                                    NegCentile = 0.3,
                                    PosCentile = 0.3,
                                    InnerModelFunc = () => SentimentUtils.CreateSvmModel()
                                };

                            case ProgramModel.NeutralZoneAuto:
                                return () => new NeutralZoneClassifier
                                {
                                    IsCalcBounds = true,
                                    InnerModelFunc = () => SentimentUtils.CreateSvmModel()
                                };

                            case ProgramModel.NeutralZoneBin:
                                return () => new NeutralZoneBinClassifier
                                {
                                    InnerModelFunc = () => SentimentUtils.CreateSvmModel(),
                                    BinWidth = 0.05
                                };

                            case ProgramModel.SvmTwoPlane:
                                return () => new TwoPlaneClassifier
                                    {
                                        InnerModelFunc = () => SentimentUtils.CreateSvmModel(),
                                        IsScorePercentile = false
                                    };

                            case ProgramModel.SvmTwoPlaneBW1:
                                return () => new TwoPlaneClassifier
                                    {
                                        InnerModelFunc = () => SentimentUtils.CreateSvmModel(),
                                        IsScorePercentile = false,
                                        BinWidth = 0.2
                                    };

                            case ProgramModel.SvmTwoPlaneBW2:
                                return () => new TwoPlaneClassifier
                                    {
                                        InnerModelFunc = () => SentimentUtils.CreateSvmModel(),
                                        IsScorePercentile = false,
                                        BinWidth = 0.1
                                    };

                            case ProgramModel.SvmTwoPlaneBW3:
                                return () => new TwoPlaneClassifier
                                    {
                                        InnerModelFunc = () => SentimentUtils.CreateSvmModel(),
                                        IsScorePercentile = true,
                                        BinWidth = 0
                                    };

                            case ProgramModel.SvmTwoPlaneBias:
                                return () => new TwoPlaneClassifier
                                {
                                    InnerModelFunc = () => SentimentUtils.CreateSvmModel(),
                                    BiasToPosRate = 0.07,
                                    BiasToNegRate = 0.04,

                                    /*
                                PosBiasCalibration = new TwoPlaneClassifier.BiasCalibration
                                {
                                    BiasLowerBound = -0.0,
                                    BiasUpperBound = 0.2,
                                    BiasStep = 0.01
                                },
                                NegBiasCalibration = new TwoPlaneClassifier.BiasCalibration
                                {
                                    BiasLowerBound = -0.0,
                                    BiasUpperBound = 0.2,
                                    BiasStep = 0.01
                                }
*/
                                };

                            case ProgramModel.SvmThreePlaneOneVsOne:
                                return () => new ThreePlaneOneVsOneClassifier
                                {
                                    InnerModelFunc = () => SentimentUtils.CreateSvmModel(),
                                };

                            case ProgramModel.SvmThreePlaneOneVsAll:
                                return () => new ThreePlaneOneVsAllClassifier
                                {
                                    InnerModelFunc = () => SentimentUtils.CreateSvmModel(),
                                };

                            case ProgramModel.SvmThreePlaneOneVsOneVoting:
                                return () => new ThreePlaneOneVsOneVotingClassifier
                                {
                                    OnCreateModel = i => SentimentUtils.CreateSvmModel(),
                                };

                            case ProgramModel.SvmThreePlaneOneVsOneBinVoting:
                                return () => new ThreePlaneOneVsOneBinVotingClassifier (0.1)
                                    {
                                        OnCreateModel = i => SentimentUtils.CreateSvmModel(),
                                    };

                            case ProgramModel.SvmThreePlaneOneVsAllVoting:
                                return () => new ThreePlaneOneVsAllVotingClassifier
                                {
                                    OnCreateModel = i => SentimentUtils.CreateSvmModel(),
                                };

                            case ProgramModel.SvmReplication:
                                return () => new ReplicationWrapperClassifier
                                {
                                    H1 = 0.8,
                                    H2 = 1,
                                    InnerModelFunc = () => SentimentUtils.CreateSvmModel()
                                };

                            case ProgramModel.NeutralZoneReliability:
                                return () => new NeutralZoneReliabilityClassifier
                                {
                                    InnerModelFunc = () => SentimentUtils.CreateSvmModel()
                                };

                            case ProgramModel.TripleSvmVoting:
                                return () => new DelegatingVotingClassifier<SentimentLabel, SparseVector<double>>(new IModel<SentimentLabel, SparseVector<double>>[3])
                                {
                                    OnCreateModel = modelIdx =>
                                    {
                                        switch (modelIdx)
                                        {
                                            case 0:
                                                return new NeutralZoneClassifier
                                                {
                                                    Centile = 0.3,
                                                    InnerModelFunc = () => SentimentUtils.CreateSvmModel()
                                                };
                                            case 1:
                                                return new TwoPlaneClassifier
                                                {
                                                    InnerModelFunc = () => SentimentUtils.CreateSvmModel()
                                                };
                                            case 2:
                                                return new ThreePlaneOneVsOneVotingClassifier
                                                {
                                                    OnCreateModel = i => SentimentUtils.CreateSvmModel()
                                                };
                                            default:
                                                throw new InvalidOperationException();
                                        }
                                    }
                                };

                            case ProgramModel.Cascading:
                                return () => new CascadingClassifier<SentimentLabel, SparseVector<double>>(
                                    new OneVsAllClassifier<SentimentLabel, SparseVector<double>>(SentimentLabel.Neutral, SentimentUtils.CreateSvmModel()), SentimentLabel.Neutral,
                                    SentimentUtils.CreateSvmModel());

                            default:
                                throw new NotSupportedException("unknown model");
                        }
                    }).ToArray();

            return modelFactory;
        }


        private static void ValidateModels(ModelsParams modelsParams, ExecutionMode executionMode, StreamWriter output, StreamWriter csvOutput)
        {
            var sw = new Stopwatch();
            sw.Start();

            output.WriteLine("\n** Performing validation...\n");
            output.Flush();

            // text preprocessing

            foreach (LabeledExample<SentimentLabel, Tweet> le in modelsParams.LabeledDataset)
            {
                le.Example.Text = modelsParams.FeatureProcessor.Run(le.Example.Text);
            }

            // perpare models

            System.Func<IModel<SentimentLabel, SparseVector<double>>>[] modelFactory = GetModelFactory(modelsParams.Model);

            // run validation -- with bow reinit

            var errPreds = new List<string>();
            

            TagDistrTable<SentimentLabel> testTagDistrTable = null;
            var validator = new BowReinitCrossValidator(modelFactory)
                {
                    Dataset = modelsParams.LabeledDataset,
                    BowSpaceFunc = modelsParams.BowSpaceFunc,
                    //IsShuffleStratified = true,
                    IsStratified = false,
                    IsShuffle = false,
                    ShuffleRandom = new Random(1),
                    ModelNameFunc = (sender, m) =>
                    {
                        string name = m.GetType().Name;
                        if (m is NeutralZoneClassifier && ((NeutralZoneClassifier)m).IsCalcBounds)
                        {
                            name += " - auto calc bounds";
                        }
                        else if (m is TwoPlaneClassifier)
                        {
                            var tpwc = (TwoPlaneClassifier)m;
                            if (tpwc.BiasToNegRate != 0 || tpwc.BiasToPosRate != 0)
                            {
                                name += string.Format(" bias to pos {0:0.00}, bias to neg {1:0.00}", tpwc.BiasToPosRate, tpwc.BiasToNegRate);
                            }
                            else if (m is TwoPlaneClassifier)
                            {
                                var tpc = (TwoPlaneClassifier)m;
                                if (tpc.BiasToNegRate != 0 || tpc.BiasToPosRate != 0)
                                {
                                    name += string.Format(" bias to pos {0:0.00}, bias to neg {1:0.00}", tpc.BiasToPosRate, tpc.BiasToNegRate);
                                }
                                if (tpc.BinWidth != 0)
                                {
                                    name += string.Format(" {0:0.000}", tpc.BinWidth);
                                }
                                if (tpc.IsScorePercentile)
                                {
                                    name += " score-percentile";
                                }
                            }
                        }
                        return name;
                    },

                    OnAfterTrain = (sender, foldN, model, trainSet) =>
                    {
                        if (model is NeutralZoneBinClassifier)
                        {
                            var nzbc = (NeutralZoneBinClassifier)model;
                            if (modelsParams.CsvOutput == CsvOutput.DistanceProbs)
                            {
                                Console.WriteLine(nzbc.TagDistrTable.ToString());
                            }
                        }
                    
                    },
                    

                    OnBeforeTest = (sender, foldN, model, testSet, mappedTestSet) =>
                    {
                        return mappedTestSet;
                    },

                    OnAfterTest = (sender, foldN, model, testSet) =>
                    {
                        Console.WriteLine("Testing completed: model {0}, fold {1}", sender.GetModelName(model), foldN);
                        if (model is TwoPlaneClassifier)
                        {
                            var tpwc = model as TwoPlaneClassifier;
                            if (tpwc.NegBiasCalibration != null || tpwc.PosBiasCalibration != null)
                            {
                                output.WriteLine("fold {0}, pos bias: {1:0.00}, neg bias: {2:0.00}, score: {3:0.00}", foldN, tpwc.BiasToPosRate,
                                    tpwc.BiasToNegRate, sender.PerfData.GetPerfMatrix(sender.ExpName, sender.GetModelName(model), foldN).GetAccuracy());
                            }
                        }
                    },

                    OnAfterValidation = sender =>
                    {
                        if (testTagDistrTable != null)
                        {
                            testTagDistrTable.Calculate();
                            OutputDistanceProbs(csvOutput, testTagDistrTable);
                        }
                    },

                    OnBeforeFold = (sender, foldN, trainSet, testSet) =>
                   {
                       int negCount = testSet.Count(p => p.Label.ToString() == "Negative");
                       int neutCount = testSet.Count(p => p.Label.ToString() == "Neutral");
                       int posCount = testSet.Count(p => p.Label.ToString() == "Positive");

                       output.WriteLine("Fold: " + foldN + "\tneg: " + negCount + "\tneut: " + neutCount + "\tpos: " + posCount);
                       Console.WriteLine("Fold: " + foldN + "\tneg: " + negCount + "\tneut: " + neutCount + "\tpos: " + posCount);
                   }

                };

            var cores = (int)(Math.Round(Environment.ProcessorCount * 0.9) - 1); // use 90% of cpu cores
            output.WriteLine("Multi-threaded using {0} cores\n", cores);
            output.Flush();

            var exceptions = new List<Exception>();

            switch (executionMode)
            {
                case ExecutionMode.ModelParallel:
                    {
                        output.WriteLine("Model level parallelization - .net");
                        output.Flush();
                        Parallel.ForEach(
                            validator.GetFoldAndModelTasks(),
                            new ParallelOptions { MaxDegreeOfParallelism = cores },
                            foldTask => Parallel.ForEach(
                                foldTask(),
                                new ParallelOptions { MaxDegreeOfParallelism = cores },
                                modelTask => modelTask()
                            )
                        );
                        break;
                    }

                case ExecutionMode.FoldParallel:
                    {
                        output.WriteLine("Fold level parallelization - .net");
                        output.Flush();
                        Parallel.ForEach(validator.GetFoldTasks(), new ParallelOptions { MaxDegreeOfParallelism = cores }, t => t());
                        break;
                    }

                case ExecutionMode.FoldParallelSmart:
                    {
                        output.WriteLine("Fold level parallelization - SmartThreadPool");
                        output.Flush();
                        var threadPool = new SmartThreadPool { /*MaxThreads = cores*/ };
                        threadPool.OnThreadInitialization += () =>
                        {
                            Thread.CurrentThread.CurrentCulture = CultureInfo.CreateSpecificCulture("en");
                        };
                        foreach (Action foldTask in validator.GetFoldTasks())
                        {
                            Action ft = foldTask;
                            threadPool.QueueWorkItem(o =>
                            {
                                ft();
                                return null;
                            }, null, wi =>
                            {
                                if (wi.Exception != null)
                                {
                                    var e = (Exception)wi.Exception;
                                    exceptions.Add(e);
                                    Console.WriteLine("{0}\n{1}", e.Message, e.StackTrace);
                                }
                            });
                        }
                        threadPool.WaitForIdle();
                        threadPool.Shutdown();
                        break;
                    }

                case ExecutionMode.ModelParallelSmart:
                    {
                        output.WriteLine("Model level parallelization - SmartThreadPool");
                        output.Flush();
                        var threadPool = new SmartThreadPool { MaxThreads = cores };
                        threadPool.OnThreadInitialization += () =>
                        {
                            Thread.CurrentThread.CurrentCulture = CultureInfo.CreateSpecificCulture("en");
                        };
                        foreach (System.Func<Action[]> foldTask in validator.GetFoldAndModelTasks())
                        {
                            System.Func<Action[]> ft = foldTask;
                            threadPool.QueueWorkItem(o =>
                            {
                                foreach (Action modelTask in ft())
                                {
                                    Action mt = modelTask;
                                    threadPool.QueueWorkItem(p =>
                                    {
                                        mt();
                                        return null;
                                    }, null, wi => { if (wi.Exception != null) { exceptions.Add((Exception)wi.Exception); } });
                                }
                                return null;
                            }, null, wi => { if (wi.Exception != null) { exceptions.Add((Exception)wi.Exception); } });
                        }
                        threadPool.WaitForIdle();
                        threadPool.Shutdown();
                        break;
                    }

                case ExecutionMode.SingleThreaded:
                    {
                        output.WriteLine("No parallelization");
                        foreach (Action foldTask in validator.GetFoldTasks())
                        {
                            foldTask();
                        }
                        break;
                    }

                default:
                    throw new NotImplementedException();
            }

            foreach (Exception exception in exceptions)
            {
                throw new Exception("Error during validation", exception);
            }

            output.WriteLine();

            OutputReport(
                modelsParams,
                output,
                modelsParams.CsvOutput == CsvOutput.Metrics ? csvOutput : null,
                validator.Models.ToArray(),
                validator.Models.Select(validator.GetModelName).ToArray(),
                validator.Models.Select(m => validator.PerfData.GetSumPerfMatrix(validator.ExpName, validator.GetModelName(m))).ToArray(),
                validator.PerfData);

            var freqWords = new Dictionary<string, List<double>>();
            foreach (KeyValuePair<int, BowSpace> kv in validator.GetBowSpaces())
            {
                if (kv.Value is SentimentDeltaBowSpace)
                {
                    var dbs = (SentimentDeltaBowSpace)kv.Value;
                    foreach (Tuple<Word, double> w in dbs.GetFreqWords())
                    {
                        List<double> freqs;
                        if (!freqWords.TryGetValue(w.Item1.MostFrequentForm, out freqs))
                        {
                            freqs = new List<double>();
                            freqWords.Add(w.Item1.MostFrequentForm, freqs);
                        }
                        freqs.Add(w.Item2);
                    }
                }
            }

            output.WriteLine("Validation duration: {0}", sw.Elapsed);
            output.Flush();

            if (csvOutput != null)
            {
                csvOutput.Flush();
                csvOutput.Close();
            }
        }


        private static void SaveModel(ModelsParams modelsParams, StreamWriter output, StreamWriter csvOutput)
        {
            Preconditions.CheckNotNull(modelsParams);
            Preconditions.CheckNotNull(modelsParams.ModelFileNameFunc);

            var sw = new Stopwatch();
            sw.Start();

            output.WriteLine("\n** Saving models {0}...\n", modelsParams.Model);
            output.Flush();

            // train and save the trained model

            foreach (ProgramModel innerModel in Enum.GetValues(typeof(ProgramModel)).Cast<ProgramModel>()
                .Where(pm => pm != ProgramModel.None && pm != ProgramModel.DeltaBowSpace && modelsParams.Model.HasFlag(pm)))
            {
                var model = new TextClassifier<SentimentLabel>
                {
                    FeatureProcessor = modelsParams.FeatureProcessor,
                    BowSpace = modelsParams.BowSpaceFunc(),
                    Model = GetModelFactory(innerModel).Single().Invoke()
                };

                if (modelsParams.IsBuildSvmPerfModel)
                {
                    model.OnTrainModel = (m, bowDataset) => BuildSvmPerfModel(modelsParams, output, bowDataset, m);
                }
                model.Train(new LabeledDataset<SentimentLabel, string>(modelsParams.LabeledDataset
                    .Select(le => new LabeledExample<SentimentLabel, string>(le.Label, le.Example.Text))));

                string modelFileName = modelsParams.ModelFileNameFunc(innerModel.ToString());
                using (var writer = new BinarySerializer(modelFileName, FileMode.Create))
                {
                    model.Save(writer);
                }

                BinTable<DisagreementCounts> disagreementBins = null;
                TagDistrTable<SentimentLabel> tagDistrTable = null;
                if (model.Model is TwoPlaneClassifier)
                {
                    tagDistrTable = ((TwoPlaneClassifier)model.Model).TagDistrTable;
                    if (tagDistrTable != null)
                    {
                        disagreementBins = new BinTable<DisagreementCounts>(
                            tagDistrTable.NumOfDimensions, tagDistrTable.BinWidth, tagDistrTable.MinValue, tagDistrTable.MaxValue);
                        if (modelsParams.CsvOutput == CsvOutput.DistanceProbs)
                        {
                            OutputDistanceProbs(csvOutput, tagDistrTable);
                        }
                    }
                }

                // load and test saved model

                using (var reader = new BinarySerializer(new FileStream(modelFileName, FileMode.Open, FileAccess.Read)))
                {
                    model = new TextClassifier<SentimentLabel>(reader);
                }

                // run the test

                var perfMatrix = new PerfMatrix<SentimentLabel>(null);
                foreach (LabeledExample<SentimentLabel, Tweet> le in modelsParams.LabeledDataset)
                {
                    Prediction<SentimentLabel> prediction = model.Predict(le.Example.Text);
                    if (prediction.Any())
                    {
                        perfMatrix.AddCount(le.Label, prediction.BestClassLabel);
                    }
                    if (disagreementBins != null && prediction is TwoPlaneClassifier.Prediction)
                    {

                        var tpp = (TwoPlaneClassifier.Prediction)prediction;
                        DisagreementCounts counts = disagreementBins[tpp.PosScore, tpp.NegScore];
                        if (counts == null)
                        {
                            counts = new DisagreementCounts();
                            disagreementBins[tpp.PosScore, tpp.NegScore] = counts;
                        }
                        switch (le.Example.DisagreementRank)
                        {
                            case 0: counts.Disagreement0Count++; break;
                            case 1: counts.Disagreement1Count++; break;
                            case 2: counts.Disagreement2Count++; break;
                        }
                    }
                }

                // output results

                if (modelsParams.CsvOutput == CsvOutput.DistanceProbsDisagr && tagDistrTable != null && disagreementBins != null)
                {
                    OutputDistanceProbs(csvOutput, tagDistrTable, disagreementBins);
                }

                output.WriteLine("******** Confusion matrix for loaded classifier on TRAINING dataset: {0}", innerModel);

                output.WriteLine("Accuracy: {0:0.000}", perfMatrix.GetAccuracy());
                output.WriteLine("{0}\n", perfMatrix);
                output.WriteLine("{0}\n", perfMatrix.ToString(new[] { ClassPerfMetric.Precision, ClassPerfMetric.Recall, ClassPerfMetric.F1 }));
                output.Flush();
            }

            output.WriteLine("Model save duration: {0}", sw.Elapsed);
            output.Flush();

            if (csvOutput != null)
            {
                csvOutput.Flush();
                csvOutput.Close();
            }
        }

        public class DisagreementCounts
        {
            public int Disagreement0Count { get; set; }
            public int Disagreement1Count { get; set; }
            public int Disagreement2Count { get; set; }
        }

        private static readonly object mOutputDistanceProbsLock = new object();

        private static void OutputDistanceProbs(StreamWriter csvOutput, TagDistrTable<SentimentLabel> tagDistrTable, BinTable<DisagreementCounts> disagreementBins = null)
        {
            csvOutput.Write("dist1\tdist2\tneg count\tneu count\tpos count\tneg prob\tneu prob\tpos prob\tbinning label");
            if (disagreementBins != null)
            {
                csvOutput.Write("\tdisagree0\tdisagree1\tdisagree2");
            }
            csvOutput.WriteLine();

            for (double d = tagDistrTable.MinValue; d <= tagDistrTable.MaxValue; d += tagDistrTable.BinWidth)
            {
                for (double e = tagDistrTable.MinValue; e <= tagDistrTable.MaxValue; e += tagDistrTable.BinWidth)
                {
                    int? neg = tagDistrTable.GetCount(SentimentLabel.Negative, d, e);
                    int? neu = tagDistrTable.GetCount(SentimentLabel.Neutral, d, e);
                    int? pos = tagDistrTable.GetCount(SentimentLabel.Positive, d, e);

                    double? pneg = tagDistrTable.GetDistrValue(SentimentLabel.Negative, d, e);
                    double? pneu = tagDistrTable.GetDistrValue(SentimentLabel.Neutral, d, e);
                    double? ppos = tagDistrTable.GetDistrValue(SentimentLabel.Positive, d, e);

                    string finalBinLabel = "";

                    var counts = new int?[] { neg, neu, pos };
                    if (counts.Sum() >= 5)
                    {
                        double?[] list = { pneg, pneu, ppos };
                        double? max = list.Max();
                        finalBinLabel = max == pneg ? "Negative" : (max == pneu ? "Neutral" : "Positive");
                    }


                    DisagreementCounts disCounts = disagreementBins != null ? disCounts = disagreementBins[d, e] : null;
                    lock (mOutputDistanceProbsLock)
                    {
                        csvOutput.Write("{6:0.00}\t{7:0.00}\t{0:0.00}\t{1:0.00}\t{2:0.00}\t{3:0.00}\t{4:0.00}\t{5:0.00}\t{8:0.00}", neg,
                            neu, pos, pneg, pneu, ppos, d, e, finalBinLabel);
                        if (disCounts != null)
                        {
                            csvOutput.Write("\t{0:0.00}\t{1:0.00}\t{2:0.00}", disCounts.Disagreement0Count, disCounts.Disagreement1Count, disCounts.Disagreement2Count);
                        }
                        csvOutput.WriteLine();
                    }
                }
            }
        }


        private static void TestModel(ModelsParams modelsParams, StreamWriter output, StreamWriter csvOutput)
        {
            Preconditions.CheckNotNull(modelsParams);
            Preconditions.CheckNotNull(modelsParams.ModelFileNameFunc);

            var sw = new Stopwatch();
            sw.Start();

            output.WriteLine("\n** Testing models {0}...\n", modelsParams.Model);
            output.Flush();

            var models = new List<IModel<SentimentLabel>>();
            var modelNames = new List<string>();
            var perfMatrices = new List<PerfMatrix<SentimentLabel>>();

            foreach (ProgramModel innerModel in Enum.GetValues(typeof(ProgramModel)).Cast<ProgramModel>()
                .Where(pm => pm != ProgramModel.None && pm != ProgramModel.DeltaBowSpace && modelsParams.Model.HasFlag(pm)))
            {
                // load saved model
                TextClassifier<SentimentLabel> model;
                string modelFileName = modelsParams.ModelFileNameFunc(innerModel.ToString());
                modelFileName = modelFileName.Replace("-test_", "-train_");

                using (var reader = new BinarySerializer(new FileStream(modelFileName, FileMode.Open, FileAccess.Read)))
                {
                    model = new TextClassifier<SentimentLabel>(reader);
                }
                if (modelsParams.BowSpaceFunc().OnGetTokens != null) // not being able to save this...
                {
                    model.BowSpace.OnGetTokens = modelsParams.BowSpaceFunc().OnGetTokens;
                }

                models.Add(model);
                modelNames.Add(innerModel.ToString());

                // run the test
                var perfMatrix = new PerfMatrix<SentimentLabel>(null);
                if (modelsParams.CsvOutput == CsvOutput.Distances)
                {
                    csvOutput.WriteLine("actual\tpredicted\tscore\tpos distance\tneg distance");
                }
                foreach (LabeledExample<SentimentLabel, Tweet> le in modelsParams.LabeledDataset)
                {
                    Prediction<SentimentLabel> prediction = model.Predict(le.Example.Text);
                    if (prediction.Any())
                    {
                        if (prediction is TwoPlaneClassifier.Prediction)
                        {
                            var tpPrediction = prediction as TwoPlaneClassifier.Prediction;
                            if (modelsParams.CsvOutput == CsvOutput.Distances)
                            {
                                csvOutput.WriteLine("{0}\t{1}\t{2:0.000}\t{3:0.000}\t{4:0.000}",
                                    le.Label, prediction.BestClassLabel, prediction.BestScore, tpPrediction.PosScore, tpPrediction.NegScore);
                            }
                        }
                        perfMatrix.AddCount(le.Label, prediction.BestClassLabel);
                    }
                }
                perfMatrices.Add(perfMatrix);
            }

            OutputReport(modelsParams, output, modelsParams.CsvOutput == CsvOutput.Metrics ? csvOutput : null,
                models.ToArray(), modelNames.ToArray(), perfMatrices.ToArray(), null);

            output.WriteLine("Model test duration: {0}", sw.Elapsed);
            output.Flush();

            if (csvOutput != null)
            {
                csvOutput.Flush();
                csvOutput.Close();
            }
        }


        private static void OutputReport(ModelsParams modelsParams, StreamWriter output, StreamWriter csvOutput,
            IModel<SentimentLabel>[] models, string[] modelNames, PerfMatrix<SentimentLabel>[] perfMatrices, PerfData<SentimentLabel> perfData)
        {
            PerfMetric[] metrics = Enum.GetValues(typeof(PerfMetric)).Cast<PerfMetric>().ToArray();
            ClassPerfMetric[] classMetrics = Enum.GetValues(typeof(ClassPerfMetric)).Cast<ClassPerfMetric>().ToArray();
            OrdinalPerfMetric[] ordinalMetrics = Enum.GetValues(typeof(OrdinalPerfMetric)).Cast<OrdinalPerfMetric>().ToArray();
            SentimentLabel[] orderedLabels = new[] { SentimentLabel.Negative, SentimentLabel.Neutral, SentimentLabel.Positive };
            var sb = new StringBuilder();
            if (csvOutput != null && csvOutput.BaseStream.Length == 0)
            {
                // header
                sb.Append("Dataset").Append("\t");
                sb.Append("Model").Append("\t");
                sb.Append("From").Append("\t");
                sb.Append("To").Append("\t");
                foreach (PerfMetric metric in metrics)
                {
                    sb.Append(metric).Append("\t");
                }
                foreach (OrdinalPerfMetric metric in ordinalMetrics)
                {
                    sb.Append(metric).Append("\t");
                }

                foreach (PerfMetric metric in metrics)
                {
                    if (metric == PerfMetric.Accuracy)
                    {
                        sb.Append(metric + " MACRO").Append("\t");
                        sb.Append("stderr (95%)").Append("\t");
                    }
                }
                foreach (OrdinalPerfMetric metric in ordinalMetrics)
                {
                    if (metric == OrdinalPerfMetric.KAlphaInterval || 
                        metric == OrdinalPerfMetric.F1AvgExtremeClasses ||
                        metric == OrdinalPerfMetric.AccuracyTolerance1)
                    {
                        sb.Append(metric + " MACRO").Append("\t");
                        sb.Append("stderr (95%)").Append("\t");
                    }
                }

                sb.AppendLine();
            }

            for (int i = 0; i < models.Length; i++)
            {
                output.WriteLine("******** Model: {0}", modelNames[i]);

                if (models[i] is TextClassifier<SentimentLabel>)
                {
                    var tc = (TextClassifier<SentimentLabel>)models[i];
                    output.WriteLine(SentimentUtils.GetObjectReport(tc.BowSpace, "Logger", "Words"));
                    output.WriteLine("Feature processor config:\n{0}", tc.FeatureProcessor);
                }

                PerfMatrix<SentimentLabel> sumPerfMatrix = perfMatrices[i];
                output.WriteLine("Accuracy: {0:0.000}", sumPerfMatrix.GetAccuracy());
                output.WriteLine("{0}\n", sumPerfMatrix);
                output.WriteLine("{0}\n", sumPerfMatrix.ToString(new[] { ClassPerfMetric.Precision, ClassPerfMetric.Recall, ClassPerfMetric.F1 }));

                if (perfData != null)
                {
                    output.WriteLine("* MACRO averaging ****");
                    string expName = perfData.GetDataKeys().First().Item1;
                    foreach (PerfMetric metric in metrics)
                    {
                        double stderr;
                        double avg = perfData.GetAvgStdErr(expName, modelNames[i], metric, out stderr);
                        output.Write(metric.ToString().PadLeft(30));
                        output.WriteLine(" {0:0.000} ± {1:0.000} ({2}%)", avg, stderr, 95);
                    }
                    output.WriteLine();
                    foreach (ClassPerfMetric metric in classMetrics)
                    {
                        foreach (SentimentLabel label in orderedLabels)
                        {
                            double stderr;
                            double avg = perfData.GetAvgStdErr(expName, modelNames[i], metric, label, out stderr);
                            output.Write((metric + " " + label).PadLeft(30));
                            output.WriteLine(" {0:0.000} ± {1:0.000} ({2}%)", avg, stderr, 95);
                        }
                    }
                    output.WriteLine();
                    foreach (OrdinalPerfMetric metric in ordinalMetrics)
                    {
                        double stderr;
                        double avg = perfData.GetAvgStdErr(expName, modelNames[i], metric, orderedLabels, out stderr);
                        output.Write(metric.ToString().PadLeft(30));
                        output.WriteLine(" {0:0.000} ± {1:0.000} ({2}%)", avg, stderr, 95);
                    }
                    output.WriteLine();
                }

                output.WriteLine("* MICRO averaging ****");
                foreach (PerfMetric metric in metrics)
                {
                    output.Write(metric.ToString().PadLeft(30));
                    output.WriteLine(" {0}", sumPerfMatrix == null ? "NaN" : sumPerfMatrix.GetScore(metric).ToString("n3"));
                }
                foreach (ClassPerfMetric metric in classMetrics)
                {
                    foreach (SentimentLabel label in orderedLabels)
                    {
                        output.Write((metric + " " + label).PadLeft(30));
                        output.WriteLine(" {0}", sumPerfMatrix == null ? "NaN" : sumPerfMatrix.GetScore(metric, label).ToString("n3"));
                    }
                }
                foreach (OrdinalPerfMetric metric in ordinalMetrics)
                {
                    output.Write(metric.ToString().PadLeft(30));
                    output.WriteLine(" {0}", sumPerfMatrix == null ? "NaN" : sumPerfMatrix.GetScore(metric, orderedLabels).ToString("n3"));
                }

                if (csvOutput != null)
                {
                    // data
                    sb.Append(modelsParams.TrainDataSource.Name).Append("\t");
                    sb.Append(modelNames[i]).Append("\t");
                    sb.Append(modelsParams.TrainDataSource.From == null ? "" : modelsParams.TrainDataSource.From.ToString()).Append("\t");
                    sb.Append(modelsParams.TrainDataSource.To == null ? "" : modelsParams.TrainDataSource.To.ToString()).Append("\t");
                    foreach (PerfMetric metric in metrics)
                    {
                        sb.Append(sumPerfMatrix.GetScore(metric).ToString("n3")).Append("\t");
                    }
                    foreach (OrdinalPerfMetric metric in ordinalMetrics)
                    {
                        sb.Append(sumPerfMatrix.GetScore(metric, orderedLabels).ToString("n3")).Append("\t");
                    }

                    if (perfData != null)
                    {
                        string expName = perfData.GetDataKeys().First().Item1;
                        foreach (PerfMetric metric in metrics)
                        {
                            if (metric == PerfMetric.Accuracy)
                            {
                                double stderr;
                                double avg = perfData.GetAvgStdErr(expName, modelNames[i], metric, out stderr);
                                sb.Append(avg.ToString("n3")).Append("\t");
                                sb.Append(stderr.ToString("n3")).Append("\t");
                            }
                        }

                        foreach (OrdinalPerfMetric metric in ordinalMetrics)
                        {
                            if (metric == OrdinalPerfMetric.KAlphaInterval ||
                                metric == OrdinalPerfMetric.F1AvgExtremeClasses ||
                                metric == OrdinalPerfMetric.AccuracyTolerance1)
                            {
                                double stderr;
                                double avg = perfData.GetAvgStdErr(expName, modelNames[i], metric, orderedLabels,
                                                                   out stderr);
                                sb.Append(avg.ToString("n3")).Append("\t");
                                sb.Append(stderr.ToString("n3")).Append("\t");
                            }
                        }

                    }


                    sb.AppendLine();
                }
            }
            if (csvOutput != null)
            {
                csvOutput.Write(sb);
                csvOutput.Flush();
            }
        }

        private static void BuildSvmPerfModel(ModelsParams modelsParams, StreamWriter output,
            LabeledDataset<SentimentLabel, SparseVector<double>> bowDataset, TextClassifier<SentimentLabel> model)
        {
            output.WriteLine("\n** Bulding SVM Perf sentiment model...\n");
            output.Flush();

            string dataSourceName = modelsParams.TrainDataSource.Name;
            string regexSearch = new string(Path.GetInvalidFileNameChars()) + new string(Path.GetInvalidPathChars());
            var r = new Regex(string.Format("[{0}]", Regex.Escape(regexSearch)));
            dataSourceName = r.Replace(dataSourceName, "").Replace(" ", "");

            string featureWeightsPath = "FeaturesWeights-" + dataSourceName + ".data";
            var trainFileFeaturesWeights = new StreamWriter(featureWeightsPath);

            foreach (LabeledExample<SentimentLabel, SparseVector<double>> example in bowDataset)
            {
                string newLine;
                if (example.Label != SentimentLabel.Neutral)
                {
                    newLine = example.Label == SentimentLabel.Positive ? "+1" : "-1";
                }
                else
                {
                    continue;
                }

                bool hasFeature = false;
                for (int j = 0; j < example.Example.Count; j++)
                {
                    double feature = example.Example.GetIdxDirect(j) + 1;
                    double weight = example.Example.GetDatDirect(j);
                    newLine += " " + feature + ":" + weight;
                    hasFeature = true;
                }

                if (hasFeature)
                {
                    trainFileFeaturesWeights.WriteLine(newLine);
                }
            }

            trainFileFeaturesWeights.Close();

            string perfModelPath = "SVMPerf_" + dataSourceName + ".model";
            TrainSvmPerfClassifier("svm_perf_learn_64.exe", string.Format("-c {0} -e 10", bowDataset.Count / 10), featureWeightsPath, perfModelPath, output);

            string[] lines = File.ReadAllLines(perfModelPath);
            lines[0] = lines[0].Replace("SVM-light Version V6.20", "SVM-light Version V6.10");
            File.WriteAllLines(perfModelPath, lines);

            var svm = new SvmBinaryClassifier<SentimentLabel>();
            svm.LoadModel(perfModelPath);

            model.Model = new NeutralZoneReliabilityClassifier(svm, bowDataset);

            File.Delete(perfModelPath);
            File.Delete(featureWeightsPath);
        }

        private static void TrainSvmPerfClassifier(string learnerPath, string learnerArguments, string trainingDataPath, string modelPath, StreamWriter trainingLog)
        {
            Console.WriteLine("Learning PerfSvm classifier...");

            var process = new Process
            {
                StartInfo =
                {
                    FileName = learnerPath,
                    Arguments = learnerArguments + " " + trainingDataPath + " " + modelPath,
                    RedirectStandardOutput = true,
                    UseShellExecute = false
                }
            };

            try
            {
                process.Start();
                using (StreamReader reader = process.StandardOutput)
                {
                    string result = reader.ReadToEnd();
                    trainingLog.WriteLine(result);
                }

            }
            catch
            {
                // Log error.
                throw;
            }
            process.WaitForExit();
        }

        public class ModelsParams
        {
            public LabeledDataset<SentimentLabel, Tweet> LabeledDataset { get; set; }
            public System.Func<BowSpace> BowSpaceFunc { get; set; }
            public ProgramModel Model { get; set; }
            public TextFeatureProcessor FeatureProcessor { get; set; }
            public LabeledTextSource TrainDataSource { get; set; }
            public LabeledTextSource TestDataSource { get; set; }
            public ConcurrentDictionary<Tuple<int, string>, ExamplePrediction> ExamplePredictions { get; set; }
            public CsvOutput CsvOutput { get; set; }
            public System.Func<string, string> ModelFileNameFunc { get; set; }
            public bool IsBuildSvmPerfModel { get; set; }
        }

        public class ExamplePrediction
        {
            public LabeledExample<SentimentLabel, SparseVector<double>> LabeledExample { get; set; }
            public ConcurrentDictionary<ProgramModel, Prediction<SentimentLabel>> ModelPredictions { get; set; }
        }
    }

    public class LabeledTextFileLabeledTextDataSource : FileLabeledTextSource
    {
        public LabeledTextFileLabeledTextDataSource(string fileName)
            : base(fileName)
        {
            Language = Language.English;
        }

        public override IEnumerable<LabeledExample<SentimentLabel, string>> GetData()
        {
            IEnumerable<string> lines = File.ReadAllLines(FileName);
            return lines.Select(l =>
            {
                string[] split = l.Split(new string[] { " " }, 2, StringSplitOptions.RemoveEmptyEntries);
                return new LabeledExample<SentimentLabel, string>(
                    (SentimentLabel)Enum.Parse(typeof(SentimentLabel), split[0]), split.Length > 1 ? split[1] : "");
            });
        }
    }
}
