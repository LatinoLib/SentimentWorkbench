using System;
using System.Linq;
using System.Collections.Generic;
using System.Data.Entity;
using System.IO;
using Latino;
using Latino.Model;
using Latino.TextMining;
using Microsoft.VisualBasic.FileIO;

namespace TextTask.DataSource
{
    public interface ILabeledTextSource<LblT> : IDisposable
    {
        string Name { get; }
        string Params { get; }
        Language Language { get; }
        
        IEnumerable<LabeledExample<LblT, string>> GetData();
        IEnumerable<LabeledExample<LblT, T>> GetData<T>();

        double DataShare { get; }
        int DataSize { get; }

        DateTime? From { get; set; }
        DateTime? To { get; set; }
    }

    public abstract class LabeledTextSource : ILabeledTextSource<SentimentLabel>
    {
        protected LabeledTextSource()
        {
            DataShare = 1;
            NumOfRetries = 5;
        }

        public abstract string Name { get; }
        public string Params { get; set; }
        public DateTime? From { get; set; }
        public DateTime? To { get; set; }
        public Language Language { get; protected set; }

        public int  NumOfRetries { get; set; }
        
        public abstract IEnumerable<LabeledExample<SentimentLabel, string>> GetData();

        public virtual IEnumerable<LabeledExample<SentimentLabel, T>> GetData<T>()
        {
            throw new NotImplementedException();
        }

        public double DataShare { get; set; }
        public virtual int DataSize { get { return 0; } }
        public int MaxDataSize { get; set; }

        public virtual void Dispose()
        {
        }
    }

    public abstract class FileLabeledTextSource : LabeledTextSource
    {
        protected FileLabeledTextSource(string fileName)
        {
            Preconditions.CheckNotNull(fileName);
            Preconditions.CheckArgument(File.Exists(fileName));

            FileName = fileName;
        }

        public string FileName { get; set; }

        public override string Name
        {
            get { return Path.GetFileNameWithoutExtension(FileName); }
        }
    }

    public abstract class CsvLabeledTextSource : FileLabeledTextSource
    {
        private int mDataSize;

        protected CsvLabeledTextSource(string fileName, string delimiters = ",")
            : base(fileName)
        {
            Delimiters = delimiters;
        }

        public string Delimiters { get; set; }
        public override int DataSize { get { return mDataSize; } }

        public override IEnumerable<LabeledExample<SentimentLabel, string>> GetData()
        {
            return GetData<Tweet>().Select(lt => new LabeledExample<SentimentLabel, string>(lt.Label, lt.Example.Text));
        }

        public override IEnumerable<LabeledExample<SentimentLabel, T>> GetData<T>()
        {
            Preconditions.CheckArgument(typeof(T) == typeof(Tweet));

            using (var parser = new TextFieldParser(FileName) { TextFieldType = FieldType.Delimited })
            {
                PrepareParser(parser);
                parser.SetDelimiters(Delimiters);
                var result = new List<LabeledExample<SentimentLabel, Tweet>>();
                while (!parser.EndOfData)
                {
                    string[] fields = parser.ReadFields();
                    SentimentLabel label;
                    Tweet tweet;
                    if (LoadLabeledTweet(fields, out label, out tweet))
                    {
                        if (tweet == null && result.Any())
                        {
                            result.Last().Label = label;
                        }
                        else
                        {
                            var labeledTweet = new LabeledExample<SentimentLabel, Tweet>(label, tweet);
                            result.Add(labeledTweet);
                        }
                    }
                }
                mDataSize = result.Count;
                return result.Cast<LabeledExample<SentimentLabel, T>>();
            }
        }

        protected abstract bool LoadLabeledTweet(string[] fields, out SentimentLabel label, out Tweet tweet);

        protected virtual void PrepareParser(TextFieldParser parser)
        {
        }
    }

    public abstract class DatabaseLabeledTextSource : LabeledTextSource
    {
        protected DatabaseLabeledTextSource(string connectionString)
        {
            DbContext = new DbContext(connectionString);
            From = DateTime.MinValue;
            To = DateTime.MaxValue;
        }

        protected DbContext DbContext { get; set; }

        public override void Dispose()
        {
            DbContext.Dispose();
            base.Dispose();
        }
    }
}