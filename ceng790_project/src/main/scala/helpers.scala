object helpers {
    def text_clean(c: Comment ) : Comment = {
        //REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        //BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        val cleanedComment = c.comment
            //.replaceAll("[/(){}\\[\\]\\|@,;]","")
            //.replaceAll("[^0-9a-zA-Z #+_]", "")
        Comment( c.label, cleanedComment, c.author, c.subreddit, c.score,
            c.ups, c.downs, c.date, c.created_utc, c.parent_comment )
    }

    def include_subreddit(c : Comment) : Comment = {
        Comment(c.label, c.comment + c.subreddit, c.author, c.subreddit, c.score,
            c.ups, c.downs, c.date, c.created_utc, c.parent_comment)
    }
}
