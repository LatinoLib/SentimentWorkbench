using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Latino;
using Latino.Model;
using Newtonsoft.Json.Linq;

namespace TextTask.DataSource
{
    public class GoldfinchDataSource : DatabaseLabeledTextSource
    {
        private static readonly object mLock = new object();

        private IEnumerable<LabeledExample<SentimentLabel, Tweet>> mData;

        public GoldfinchDataSource(string connectionString)
            : base(connectionString)
        {
            JArray duc = JArray.Parse(File.ReadAllText(@"data\sowa_duc.json"));
            AvailableEntities = duc
                .Children<JObject>().Select(obj => new Entity
                    {
                        Name = obj.Value<string>("entityName"),
                        ClassifierId = obj.Value<string>("classifierId"),
                        Enabled = obj.Value<bool>("enabled"),
                        Focuses = obj.GetValue("focuses") == null
                            ? new string[0]
                            : obj.GetValue("focuses").Values<string>().ToArray(),
                        Projects = obj.GetValue("projects") == null
                            ? new string[0]
                            : obj.GetValue("projects").Values<string>().ToArray()
                    }).ToArray();

            Entities = new List<Entity>();
        }

        public Entity[] AvailableEntities { get; private set; }
        public List<Entity> Entities { get; private set; }
        public string DomainName { get; set; }

        public override int DataSize { get { return mData == null ? 0 : mData.Count(); } }

        public override string Name
        {
            get
            {
                string[] selectedProjects = Entities.Where(e => e.Selected).SelectMany(e => e.Projects).Select(p => "'" + p + "'").ToArray();
                return string.Join(" ", selectedProjects);
            }
        }

        public override IEnumerable<LabeledExample<SentimentLabel, string>> GetData()
        {
            return GetData<Tweet>().Select(lt => new LabeledExample<SentimentLabel, string>(lt.Label, lt.Example.Text));
        }

        public override IEnumerable<LabeledExample<SentimentLabel, T>> GetData<T>()
        {
            Preconditions.CheckArgument(typeof(T) == typeof(Tweet));
            Preconditions.CheckArgumentRange(DataShare > 0 && DataShare <= 1);

            if (mData != null)
            {
                return mData.Cast<LabeledExample<SentimentLabel, T>>();
            }

            string[] selectedFocuses = Entities
                .Where(e => e.Selected).SelectMany(e => e.Focuses).Select(f => "'" + f + "'").ToArray();
            string focusTerm = selectedFocuses.Any() ? string.Format("and f.IdStr in ({0})", string.Join(",", selectedFocuses)) : "";

            string[] selectedProjects = Entities
                .Where(e => e.Selected).SelectMany(e => e.Projects).Select(p => "'" + p + "'").ToArray();
            string projectTerm = selectedProjects.Any() ? string.Format("and p.IdStr in ({0})", string.Join(",", selectedProjects)) : "";

            string domainTerm = string.IsNullOrEmpty(DomainName) ? "" : string.Format(" AND d.Name = '{0}'", DomainName);

            string sqlFromClause = string.Format(@"
                FROM   unit u
                       JOIN TweetUnit tu
                         ON tu.id = u.id
                       JOIN labelunit lu
                         ON lu.unitid = u.id
                       JOIN label l
                         ON l.id = lu.labelid AND l.overridenbyid IS NULL
                       JOIN focus f
                         ON f.id = u.focusid {0}
                       JOIN project p
                         ON p.id = f.ProjectId {1}
                      JOIN domainproject dp
                         ON p.id = dp.Project_Id 
                      JOIN domain d
                         ON d.id = dp.Domain_Id {2}
                WHERE u.DataTime >= @p0 AND u.DataTime < @p1 AND l.name in ('Negative', 'Neutral', 'Positive')
                ORDER  BY NEWID()
            ", focusTerm, projectTerm, domainTerm);

            string topTerm = "";
            if (DataShare < 1)
            {
                lock (mLock)
                {
                    int count = DbContext.Database.SqlQuery<int>("SELECT COUNT(*) " + sqlFromClause, From, To).Single();
                    topTerm = string.Format("TOP {0}", (int)Math.Truncate(DataShare * count));
                }
            }

            string sqlSelectClause = string.Format(@"
                SELECT {0} /*tu.TwitterId Id, u.DataTime Date, tu.UserScreenName [User],*/ tu.Text, 
                    CASE l.Name when 'Negative' THEN -1  when 'Neutral' THEN 0 when 'Positive' THEN 1 END Label
            ", topTerm);

            lock (mLock)
            {
                mData = DbContext.Database.SqlQuery<LabeledTweet>(sqlSelectClause + sqlFromClause, From, To)
                    .Select(lt => new LabeledExample<SentimentLabel, Tweet>(lt.Label, lt));
            }

            return mData.Cast<LabeledExample<SentimentLabel, T>>();
        }

        public override void Dispose()
        {
            mData = null;
            base.Dispose();
        }

        public class Entity
        {
            public string Name { get; set; }
            public bool Enabled { get; set; }
            public string ClassifierId { get; set; }
            public string[] Focuses { get; set; }
            public string[] Projects { get; set; }

            public bool Selected { get; set; }
        }

        private class LabeledTweet : Tweet
        {
            public SentimentLabel Label { get; set; }
        }
    }
}