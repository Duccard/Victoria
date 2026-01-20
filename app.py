# 6. THE REFINED CHAT LOGIC
if prompt := st.chat_input("Ask about the Victorian Era..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Simple Greeting Check
        greetings = [
            "hello",
            "hi",
            "greetings",
            "good morning",
            "good afternoon",
            "how are you",
        ]

        if prompt.lower().strip() in greetings:
            response_text = "Good day to you! I am Victoria. How may I assist your historical research into the Victorian Era today?"
            st.markdown(response_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )

        else:
            # 2. Historical Research Logic (Only runs if it's not a greeting)
            with st.spinner("Searching the Royal Archives..."):
                try:
                    response = victoria_brain.invoke({"query": prompt})
                    answer = response["result"]
                    st.markdown(answer)

                    # Only show the expander if documents were actually found
                    if response.get("source_documents"):
                        with st.expander("📜 View Historical Evidence"):
                            citations = set()
                            for doc in response["source_documents"]:
                                source_name = os.path.basename(
                                    doc.metadata.get("source", "Archive")
                                )
                                page = doc.metadata.get("page", "N/A")
                                if isinstance(page, int):
                                    page += 1
                                citations.add(
                                    f"**Source:** {source_name} (Page {page})"
                                )

                            for citation in sorted(citations):
                                st.markdown(f"* {citation}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error("The telegraph lines are down.")
