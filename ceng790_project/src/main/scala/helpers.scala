import org.apache.spark.sql.Dataset

object helpers {

    def text_clean(c: Comment ) : Comment = {
        //REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        //BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        val cleanedComment = c.comment
            .toLowerCase
            .trim
            /*.replaceAll("[/(){}\\[\\]\\|@,;!]"," ")
            .replaceAll("[^a-zA-Z _]", "")
            .toLowerCase
                .trim
            .replaceAll("( {1,})", " ")*/
        Comment( c.label, cleanedComment, c.author, c.subreddit, c.score,
            c.ups, c.downs, c.date, c.created_utc, c.parent_comment )
    }

    def includeSubreddit(c : Comment) : Comment = {
        Comment(c.label, c.comment + c.subreddit, c.author, c.subreddit, c.score,
            c.ups, c.downs, c.date, c.created_utc, c.parent_comment)
    }

    def includeParentComment(c : Comment) : Comment = {
        Comment(c.label, c.comment + c.parent_comment, c.author, c.subreddit, c.score,
            c.ups, c.downs, c.date, c.created_utc, c.parent_comment)
    }

    def includeAuthor(c : Comment) : Comment = {
        Comment(c.label, c.comment + c.author, c.author, c.subreddit, c.score,
            c.ups, c.downs, c.date, c.created_utc, c.parent_comment)
    }

    def countHelper(count:Long, sarcastics:Long): Unit = {
        println("Total Number of Comments: ", count)
        println("Number of Sarcastic Comments: ", sarcastics)
        println("Number of Non-Sarcastic Comments: ", count - sarcastics)
    }

    def countSarcastics(comments : Dataset[Comment]) : Unit = {
        val count = comments.count()
        val sarcastics = comments.filter(c => c.label == 1).count()

        comments.filter(c => c.label == 0).show(75, false)
        comments.filter(c => c.label == 1).show(75, false)

        countHelper(count, sarcastics)

        //val subreddits = comments.groupBy("subreddit")

    }
}
