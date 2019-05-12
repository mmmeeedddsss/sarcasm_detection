case class Comment
(
    label: Option[Int], comment: String, author: String, subreddit: String, score: Option[Int],
    ups: Option[Int], downs: Option[Int], date: String, created_utc: String, parent_comment: String
)