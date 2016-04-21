using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Latino;
using Latino.Model.Eval;

namespace TextTask
{     
    public class TaskReport
    {
        public StreamWriter Writer { get; set; }
        public delegate void MakeHandler(Task task);
        public MakeHandler OnMake { get; set; }

        public virtual void Make(Task task)
        {
            if (OnMake != null)
            {
                OnMake(task);
                return;
            }
            throw new NotImplementedException();
        }
    }

    public class ConsoleReport : TaskReport
    {
        public ConsoleReport()
        {
            Writer = new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true };
        }
    }

    public class ErrorReport : TaskReport
    {
        public ErrorReport(StreamWriter writer)
        {
            Writer = Preconditions.CheckNotNull(writer);
        }

        public Exception Exception { get; set; }

        public override void Make(Task task)
        {
            if (Exception == null) { return; }

            Writer.Write(DateTime.Now + " " + Exception.GetType());
            Writer.WriteLine(Exception.Message);
            Writer.WriteLine(Exception.StackTrace);
            Writer.Flush();
        }
    }

    public class TestTabReport : TaskReport
    {
        private readonly List<Tuple<string, Func<string>>> mExtraFields = new List<Tuple<string, Func<string>>>();

        public TestTabReport(StreamWriter writer)
        {
            Metrics = Enum.GetValues(typeof(PerfMetric)).Cast<PerfMetric>().ToArray();
            ClassMetrics = Enum.GetValues(typeof(ClassPerfMetric)).Cast<ClassPerfMetric>().ToArray();
            OrdinalMetrics = Enum.GetValues(typeof(OrdinalPerfMetric)).Cast<OrdinalPerfMetric>().ToArray();
            OrderedLabels = new[] { SentimentLabel.Negative, SentimentLabel.Neutral, SentimentLabel.Positive };

            Writer = Preconditions.CheckNotNull(writer);
        }

        public PerfMetric[] Metrics { get; private set; }
        public ClassPerfMetric[] ClassMetrics { get; private set; }
        public OrdinalPerfMetric[] OrdinalMetrics { get; private set; }
        public SentimentLabel[] OrderedLabels { get; private set; }

        public TestTabReport ExtraField(string name, Func<string> valueFunc)
        {
            Preconditions.CheckNotNull(name);
            Preconditions.CheckNotNull(valueFunc);
            mExtraFields.Add(new Tuple<string, Func<string>>(name, valueFunc));
            return this;
        }

        public override void Make(Task task)
        {
            if (Writer != null)
            {
                var sb = new StringBuilder();
                if (Writer.BaseStream.Length == 0)
                {
                    // write header
                    sb.Append("Experiment").Append("\t");
                    sb.Append("Task").Append("\t");
                    sb.Append("Dataset").Append("\t");
                    sb.Append("Volume").Append("\t");
                    sb.Append("Filtered Volume").Append("\t");
                    sb.Append("Model").Append("\t");
                    sb.Append("From").Append("\t");
                    sb.Append("To").Append("\t");
                    sb.Append("Interval Days").Append("\t");
                    sb.Append("Performance Millis").Append("\t");
                    foreach (SentimentLabel label in OrderedLabels)
                    {
                        sb.Append(label).Append(" Actual").Append("\t");
                        sb.Append(label).Append(" Predicted").Append("\t");
                    }
                    foreach (PerfMetric metric in Metrics)
                    {
                        sb.Append(metric).Append("\t");
                    }
                    foreach (ClassPerfMetric metric in ClassMetrics)
                    {
                        foreach (SentimentLabel label in OrderedLabels)
                        {
                            sb.Append(metric).Append(" - ").Append(label).Append("\t");
                        }
                    }
                    foreach (OrdinalPerfMetric metric in OrdinalMetrics)
                    {
                        sb.Append(metric).Append("\t");
                    }

                    foreach (Tuple<string, Func<string>> field in mExtraFields)
                    {
                        sb.Append(field.Item1).Append("\t");
                    }
                    sb.AppendLine();
                    lock (Writer)
                    {
                        Writer.Write(sb);
                        Writer.Flush();
                    }
                    sb = new StringBuilder();
                }

                TaskContext ctx = task.Context;
                for (int i = 0; i < ctx.Models.Length; i++)
                {
                    PerfMatrix<SentimentLabel> perfMatrix = task.PerfData.GetPerfMatrix(task.ExperimentName, task.Context.GetModelName(i), 1);

                    sb.Append(task.ExperimentName ?? "").Append("\t");
                    sb.Append(task.Name ?? "").Append("\t");
                    sb.Append(ctx.DataSource.Name).Append("\t");
                    sb.Append(ctx.DataSource.DataSize).Append("\t");
                    sb.Append(perfMatrix == null ? "NaN" : perfMatrix.GetSumAll().ToString("d")).Append("\t");
                    sb.Append(ctx.GetModelName(i)).Append("\t");
                    sb.Append(ctx.DataSource.From == null ? "" : ctx.DataSource.From.Value.ToString("yyyy-MM-dd hh:mm:ss")).Append("\t");
                    sb.Append(ctx.DataSource.To == null ? "" : ctx.DataSource.To.Value.ToString("yyyy-MM-dd hh:mm:ss")).Append("\t");
                    sb.Append(ctx.DataSource.From == null || ctx.DataSource.To == null ? "0" : 
                        (ctx.DataSource.To.Value - ctx.DataSource.From.Value).TotalDays.ToString("f1")).Append("\t");
                    sb.Append((task.PerformDuration ?? TimeSpan.Zero).TotalMilliseconds.ToString("f1")).Append("\t");


                    if (perfMatrix != null)
                    {
                        perfMatrix.AddLabels(OrderedLabels); // some ordered metrics require this 
                    } 

                    foreach (SentimentLabel label in OrderedLabels)
                    {
                        sb.Append(perfMatrix == null ? "NaN" : perfMatrix.GetActual(label).ToString("d")).Append("\t");
                        sb.Append(perfMatrix == null ? "NaN" : perfMatrix.GetPredicted(label).ToString("d")).Append("\t");
                    }
                    foreach (PerfMetric metric in Metrics)
                    {
                        sb.Append(perfMatrix == null ? "NaN" : perfMatrix.GetScore(metric).ToString("n3")).Append("\t");
                    }
                    foreach (ClassPerfMetric metric in ClassMetrics)
                    {
                        foreach (SentimentLabel label in OrderedLabels)
                        {
                            sb.Append(perfMatrix == null ? "NaN" : perfMatrix.GetScore(metric, label).ToString("n3")).Append("\t");
                        }
                    }
                    foreach (OrdinalPerfMetric metric in OrdinalMetrics)
                    {
                        sb.Append(perfMatrix == null ? "NaN" : perfMatrix.GetScore(metric, OrderedLabels).ToString("n3")).Append("\t");
                    }

                    foreach (Tuple<string, Func<string>> field in mExtraFields)
                    {
                        sb.Append(field.Item2()).Append("\t");
                    }
                    sb.AppendLine();
                }
                lock (Writer)
                {
                    Writer.Write(sb);
                    Writer.Flush();
                }
            }
        }
    }

    public class TestAgregTabReport : TaskReport
    {
        private readonly List<Tuple<string, Func<string>>> mExtraFields = new List<Tuple<string, Func<string>>>();

        public TestAgregTabReport(StreamWriter writer)
        {
            Metrics = Enum.GetValues(typeof(PerfMetric)).Cast<PerfMetric>().ToArray();
            ClassMetrics = Enum.GetValues(typeof(ClassPerfMetric)).Cast<ClassPerfMetric>().ToArray();
            OrdinalMetrics = Enum.GetValues(typeof(OrdinalPerfMetric)).Cast<OrdinalPerfMetric>().ToArray();
            OrderedLabels = new[] { SentimentLabel.Negative, SentimentLabel.Neutral, SentimentLabel.Positive };

            Writer = Preconditions.CheckNotNull(writer);
        }

        public PerfMetric[] Metrics { get; private set; }
        public ClassPerfMetric[] ClassMetrics { get; private set; }
        public OrdinalPerfMetric[] OrdinalMetrics { get; private set; }
        public SentimentLabel[] OrderedLabels { get; private set; }

        public TestAgregTabReport ExtraField(string name, Func<string> valueFunc)
        {
            Preconditions.CheckNotNull(name);
            Preconditions.CheckNotNull(valueFunc);
            mExtraFields.Add(new Tuple<string, Func<string>>(name, valueFunc));
            return this;
        }

        public override void Make(Task task)
        {
            if (Writer != null && task.PerfData != null)
            {
                // group matrices by the test case
                Tuple<string, string, string[]>[] keys = task.PerfData.GetDataKeys()
                    .GroupBy(k => string.Join("\t", k.Item1.Split('\t').Take(5)))
                    .SelectMany(g => g
                        .Select(t => t.Item2).Distinct()
                        .Select(item2 => new Tuple<string, string, string[]>(
                            g.Key, 
                            item2, 
                            g.Select(t => t.Item1.Split('\t').Skip(5).FirstOrDefault())
                                .Where(k => !string.IsNullOrEmpty(k)).Distinct().ToArray())))
                    .ToArray();

                var sb = new StringBuilder();
                if (Writer.BaseStream.Length == 0)
                {
                    // write header
                    sb.Append("Experiment").Append("\t");
                    sb.Append("Task").Append("\t");
                    sb.Append("Train Data").Append("\t");
                    sb.Append("Test Data").Append("\t");
                    sb.Append("Subsampling").Append("\t");
                    sb.Append("Train-Test Gap").Append("\t");
                    sb.Append("Train Span Days").Append("\t");
                    sb.Append("Model").Append("\t");
                    sb.Append("Performance Millis").Append("\t");
                    foreach (SentimentLabel label in OrderedLabels)
                    {
                        sb.Append(label).Append(" Actual").Append("\t");
                        sb.Append(label).Append(" Predicted").Append("\t");
                    }
                    foreach (PerfMetric metric in Metrics)
                    {
                        sb.Append(metric).Append("\t");
                        sb.Append(metric).Append(" - StdErr 90%\t");
                        sb.Append(metric).Append(" - StdErr 95%\t");
                        sb.Append(metric).Append(" - StdErr 99%\t");
                    }
                    foreach (ClassPerfMetric metric in ClassMetrics)
                    {
                        foreach (SentimentLabel label in OrderedLabels)
                        {
                            sb.Append(metric).Append(" - ").Append(label).Append("\t");
                            sb.Append(metric).Append(" - ").Append(label).Append(" - StdErr 90%\t");
                            sb.Append(metric).Append(" - ").Append(label).Append(" - StdErr 95%\t");
                            sb.Append(metric).Append(" - ").Append(label).Append(" - StdErr 99%\t");
                        }
                    }
                    foreach (OrdinalPerfMetric metric in OrdinalMetrics)
                    {
                        sb.Append(metric).Append("\t");
                        sb.Append(metric).Append(" - StdErr 90%\t");
                        sb.Append(metric).Append(" - StdErr 95%\t");
                        sb.Append(metric).Append(" - StdErr 99%\t");
                    }

                    if (keys.Any(k => k.Item3.Length > 0))
                    {
                        foreach (string metric in new[]
                            {
                                "Agreement 0", "Agreement 1", "Agreement 2", "Agreement 3", "Agreement 4", "Agreement 5", 
                                "Disagreement 0", "Disagreement 1", "Disagreement 2"
                            })
                        {
                            sb.Append(metric).Append("\t");
                            sb.Append(metric).Append(" - StdErr 90%\t");
                            sb.Append(metric).Append(" - StdErr 95%\t");
                            sb.Append(metric).Append(" - StdErr 99%\t");
                        }
                    }

                    foreach (Tuple<string, Func<string>> field in mExtraFields)
                    {
                        sb.Append(field.Item1).Append("\t");
                    }
                    sb.AppendLine();
                    lock (Writer)
                    {
                        Writer.Write(sb);
                        Writer.Flush();
                    }
                    sb = new StringBuilder();
                }

                foreach (Tuple<string, string, string[]> key in keys)
                {
                    try
                    {
                        string[] split = key.Item1.Split('\t');
                        string trainSource = split[0];
                        string testSource = split[1];
                        string spanDays = split[2];
                        string share = split[3];
                        string gap = split[4];
                        string modelName = key.Item2;

                        sb.Append(task.ExperimentName ?? "").Append("\t");
                        sb.Append(task.Name ?? "").Append("\t");
                        sb.Append(trainSource ?? "").Append("\t");
                        sb.Append(testSource ?? "").Append("\t");
                        sb.Append(share ?? "").Append("\t");
                        sb.Append(gap ?? "").Append("\t");
                        sb.Append(spanDays ?? "").Append("\t");
                        sb.Append(modelName).Append("\t");
                        sb.Append((task.PerformDuration ?? TimeSpan.Zero).TotalMilliseconds.ToString("f1")).Append("\t");

                        PerfMatrix<SentimentLabel> sumPerfMatrix;
                        try
                        {
                            sumPerfMatrix = task.PerfData.GetSumPerfMatrix(key.Item1, modelName);
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine(e);
                            sumPerfMatrix = null;
                        }
                        if (sumPerfMatrix != null)
                        {
                            sumPerfMatrix.AddLabels(OrderedLabels); // some ordered metrics require this 
                        } 
                        foreach (SentimentLabel label in OrderedLabels)
                        {
                            sb.Append(sumPerfMatrix == null ? "NaN" : sumPerfMatrix.GetActual(label).ToString("d")).Append("\t");
                            sb.Append(sumPerfMatrix == null ? "NaN" : sumPerfMatrix.GetPredicted(label).ToString("d")).Append("\t");
                        }

                        double stderr90, stderr95, stderr99;
                        foreach (PerfMetric metric in Metrics)
                        {
                            double avg = task.PerfData.GetAvgStdErr(key.Item1, modelName, metric, out stderr90, 0.90);
                            task.PerfData.GetAvgStdErr(key.Item1, modelName, metric, out stderr95, 0.95);
                            task.PerfData.GetAvgStdErr(key.Item1, modelName, metric, out stderr99, 0.99);
                        
                            sb.Append(sumPerfMatrix == null ? "NaN" : avg.ToString("n3")).Append("\t");
                            sb.Append(sumPerfMatrix == null ? "NaN" : stderr90.ToString("n3")).Append("\t");
                            sb.Append(sumPerfMatrix == null ? "NaN" : stderr95.ToString("n3")).Append("\t");
                            sb.Append(sumPerfMatrix == null ? "NaN" : stderr99.ToString("n3")).Append("\t");
                        }
                        foreach (ClassPerfMetric metric in ClassMetrics)
                        {
                            foreach (SentimentLabel label in OrderedLabels)
                            {
                                double avg = task.PerfData.GetAvgStdErr(key.Item1, modelName, metric, label, out stderr90, 0.90);
                                task.PerfData.GetAvgStdErr(key.Item1, modelName, metric, label, out stderr95, 0.95);
                                task.PerfData.GetAvgStdErr(key.Item1, modelName, metric, label, out stderr99, 0.99);

                                sb.Append(sumPerfMatrix == null ? "NaN" : avg.ToString("n3")).Append("\t");
                                sb.Append(sumPerfMatrix == null ? "NaN" : stderr90.ToString("n3")).Append("\t");
                                sb.Append(sumPerfMatrix == null ? "NaN" : stderr95.ToString("n3")).Append("\t");
                                sb.Append(sumPerfMatrix == null ? "NaN" : stderr99.ToString("n3")).Append("\t");
                            }
                        }
                        foreach (OrdinalPerfMetric metric in OrdinalMetrics)
                        {
                            double avg = task.PerfData.GetAvgStdErr(key.Item1, modelName, metric, OrderedLabels, out stderr90, 0.90);
                            task.PerfData.GetAvgStdErr(key.Item1, modelName, metric, OrderedLabels, out stderr95, 0.95);
                            task.PerfData.GetAvgStdErr(key.Item1, modelName, metric, OrderedLabels, out stderr99, 0.99);

                            sb.Append(sumPerfMatrix == null ? "NaN" : avg.ToString("n3")).Append("\t");
                            sb.Append(sumPerfMatrix == null ? "NaN" : stderr90.ToString("n3")).Append("\t");
                            sb.Append(sumPerfMatrix == null ? "NaN" : stderr95.ToString("n3")).Append("\t");
                            sb.Append(sumPerfMatrix == null ? "NaN" : stderr99.ToString("n3")).Append("\t");
                        }

                        // extract agreement data
                        if (key.Item3.Length > 0)
                        {
                            foreach (string agreementKey in new[]
                                {
                                    "Agreement 0", "Agreement 1", "Agreement 2", "Agreement 3", "Agreement 4", "Agreement 5", 
                                    "Disagreement 0", "Disagreement 1", "Disagreement 2"
                                })
                            {
                                string matrixKey = key.Item1 + "\t" + key.Item3.FirstOrDefault(k => k.EndsWith(agreementKey));
                                if (matrixKey == null)
                                {
                                    sb.Append("\t\t\t\t");
                                    continue;
                                }

                                double avg = task.PerfData.GetAvgStdErr(matrixKey, modelName, ClassPerfMetric.ActualCount, SentimentLabel.Exclude, out stderr90, 0.90);
                                task.PerfData.GetAvgStdErr(matrixKey, modelName, ClassPerfMetric.ActualCount, SentimentLabel.Exclude, out stderr95, 0.95);
                                task.PerfData.GetAvgStdErr(matrixKey, modelName, ClassPerfMetric.ActualCount, SentimentLabel.Exclude, out stderr99, 0.99);

                                sb.Append(sumPerfMatrix == null ? "NaN" : avg.ToString("n3")).Append("\t");
                                sb.Append(sumPerfMatrix == null ? "NaN" : stderr90.ToString("n3")).Append("\t");
                                sb.Append(sumPerfMatrix == null ? "NaN" : stderr95.ToString("n3")).Append("\t");
                                sb.Append(sumPerfMatrix == null ? "NaN" : stderr99.ToString("n3")).Append("\t");
                            }
                        }

                        foreach (Tuple<string, Func<string>> field in mExtraFields)
                        {
                            sb.Append(field.Item2()).Append("\t");
                        }
                        sb.AppendLine();
                    }
                    catch (Exception e)
                    {
                        sb.Append("error: " + e.Message);
                        sb.AppendLine();
                        task.MakeErrorReport(e);
                    }
                }

                lock (Writer)
                {
                    Writer.Write(sb);
                    Writer.Flush();
                }
            }
        }
    }
}