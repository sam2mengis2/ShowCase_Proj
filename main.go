package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
)

const LIMIT uint8 = 100

func main() {
	http.HandleFunc("/", page)

	if err := http.ListenAndServe(":3000", nil); err != nil {
		log.Fatal(err)
	}
}

func page(w http.ResponseWriter, r *http.Request){
	file_content, err := loadPage("/home/ygg/projects/showcase/public/index.html")
	if err != nil {
		fmt.Fprintf(w,"Error Could not load page!")
	}

	fmt.Fprintf(w,"%s", file_content)
}


func loadPage(filepath string) ([]byte, error) {
	content, err := os.ReadFile(filepath)
	if err != nil {
		return nil, err
	}

	return content, nil
}




